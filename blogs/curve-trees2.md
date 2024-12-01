---
title: Curve Trees - Bulletproofs And Algorithms
<!-- bibliography: ../assets/curve-trees2.bib -->
link-citations: true
cite-method: biblatex
---

In [the previous post](/blogs/curve-trees1.html) I introduced the general architecture of curve trees, but it isn't complete without understanding how Bulletproofs work.

In a few words, the Bulletproofs paper introduced an improved inner product argument, proving that you know two vectors $\vec{A}$ and $\vec{B}$ such that a) the commitment $Com(\vec{A}, \vec{B}) = P$ for some public point $P$ and b) the inner product $\langle \vec{A}, \vec{B} \rangle = c$ for some public scalar $c$. It builds over previous iterations of the same 'framework', which let you efficiently encode problems (namely, arithmetic circuits) into this inner product form.
