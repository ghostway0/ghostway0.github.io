---
title: 'The UnslothAI challenge: DeepSeek and optimization'
---

I, like many others, found [the challenge](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH?usp=sharing) on Twitter ($\mathbb{X}$) and quite immediately found them interesting. They totally were, but not for the reasons I had imagined.

I attempted the first challenge -- writing a dequantization kernel for NF4->other floats. [The `fast_dequantize` function](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py#L128) from UnslothAI's library is the starting point:

```py
ptr_out_absmax = get_ptr(out_absmax)
cdequantize_blockwise_fp32(
    get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
    ctypes_c_int(blocksize2), ctypes_c_int(n_elements_absmax),
)
out_absmax += offset

fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
     cdequantize_blockwise_bf16_nf4
fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
   ctypes_c_int(blocksize), ctypes_c_int(out.numel()),)
```

`cdequantize_blockwise_fp16_nf4` and `cdequantize_blockwise_bf16_nf4` are instantiations of [`kDequantizeBlockwise`](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu#L595) and, essentially, they're decompressing the statistics (absmax) in blocks using absmax2 and a lookup table (code)--similar to PNG's palletes--and then multiplying dequantized NF4 floats by those decompressed absmax. After two days full of trial, reversing and errors, I managed to write something that works:

```py
@triton.jit
def _your_dequantize_nf4_kernel(
    code_ptr: tl.tensor,
    a_ptr: tl.tensor,
    absmax_ptr: tl.tensor,  # compressed absmax
    absmax2_ptr: tl.tensor, # absmax of the absmax
    out_ptr: tl.tensor,
    blocksize: tl.constexpr,
    n_elements: tl.constexpr,
    TILE_SIZE: tl.constexpr, # processes 2 * TILE_SIZE elements
    absmax_blocksize: tl.constexpr,
    absmax_nelems: tl.constexpr,
    absmax_offset: tl.constexpr,
    lookup_ptr: tl.tensor,
):
    pid_m = tl.program_id(0)
    base_idx = pid_m * TILE_SIZE

    base_offsets = base_idx + tl.arange(0, TILE_SIZE)

    absmax = tl.load(absmax2_ptr + base_offsets // (absmax_blocksize * blocksize))
    absmax_bytes = tl.load(absmax_ptr + base_offsets // blocksize)
    local_abs_max = tl.load(code_ptr + absmax_bytes) * absmax + absmax_offset

    qvals_bytes = tl.load(a_ptr + base_offsets, mask=base_offsets < n_elements // 2, other=0)

    first_nibble  = qvals_bytes & 0b1111
    second_nibble = (qvals_bytes >> 4) & 0b1111

    val0 = tl.load(lookup_ptr + first_nibble) * local_abs_max
    val1 = tl.load(lookup_ptr + second_nibble) * local_abs_max

    even_offsets = base_offsets * 2
    odd_offsets = even_offsets + 1

    tl.store(out_ptr + odd_offsets, val0, mask=odd_offsets < n_elements)
    tl.store(out_ptr + even_offsets, val1, mask=even_offsets < n_elements)
```

Why did those 20 lines of code take two sleep-deprived days to write? Well, that's today's story.

// terrible code-reading skills when I didn't understand the two calls to dequantize (just did the nf4 dequantize)
My terrible code-reading skills weren't the only problem, though: in the original challenge's code, `torch.set_default_dtype(test_dtype)` was called, making some down-the-line code generate the `code` tensor with that dtype. You may have noticed that Unsloth's code did not check any of that, nor did it check if the test dtype is included in their \[fp16, bf16\] expected dtypes.

Let's take another look at the Cuda kernel:

```cpp
template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(float *code, unsigned char * A, float * absmax, T *out, const int blocksize, const int n)
{

  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1)];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH*((DATA_TYPE > 0) ? 2 : 1), cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE) // a loop that only executes once
  {
    if (DATA_TYPE > 0)
    {
      valid_items_load = min(TILE_SIZE, (n + 1) / 2 - i);
      valid_items_store = min(TILE_SIZE * 2, n - i * 2);
    }
    else
    {
      valid_items_load = min(TILE_SIZE, n - i);
      valid_items_store = valid_items_load;
    }

    // Since blocksize will always be a power-of-2, we avoid more expensive
    // division by the blocksize and instead use a shift operation.
    // This is equivalent to (i+threadId.x*NUM_PER_TH)/blocksize.
    local_abs_max = __ldg(&absmax[(i+threadIdx.x*NUM_PER_TH) >> (31 - __clz(blocksize))]);

    __syncthreads();
    LoadChar(loadchar).Load(&(A[i]), qvals, valid_items_load, 128);

    switch (DATA_TYPE)
    {
        case General8bit:
          // load code through read-only cache via __ldg
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++) // if NUM_PER_TH is larger than the block size, this would use the wrong local_abs_max
            vals[j] = __ldg(&code[qvals[j]])*local_abs_max;
          break;
        case FP4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
            vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
          }
          break;
        case NF4:
          #pragma unroll NUM_PER_TH
          for(int j = 0; j < NUM_PER_TH; j++)
          {
            vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
            vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max; // why is the lower part on the higher address?
          }
          break;
    }

    __syncthreads();
    StoreT(storet).Store(&(out[(DATA_TYPE > 0) ? i*2 : i]), vals, valid_items_store);
  }
}
```



Code was fp16 because of the set_default_, but no checks, raw pointer
fp16/bf16 no check for inclusion
cuda kernel for loop only once
per_thread weird and could be wrong -> assumptions about block size
