---
title: Cards
---

<style>
#card-details, #loading-message, #more-link {
  color: #00ff00;
  font-family: 'Courier New', Courier, monospace;
}

#loading-message {
  margin-bottom: 20px;
}

#spinner {
  display: inline-block;
  width: 1ch;
  text-align: center;
  animation: spin-text 1s steps(4, end) infinite;
  margin: 0;
  color: #00ff00;
}

@keyframes spin-text {
  0% { content: '|'; }
  25% { content: '/'; }
  50% { content: '-'; }
  75% { content: '\\'; }
  100% { content: '|'; }
}

.prompt {
  color: #00ff00;
}
</style>

<div id="loading-message"></div>
<div id="spinner"></div>
<div id="card-details"></div>
<div id="card-details">
</div>

<a href="#" id="more">[More]</a>

<script>

const CARD_HOLDERS = [
  // celebrities
  // yes I know these are not your typical celebrities thank you very much
  'Linus Torvalds', 'Ada Lovelace', 'Alan Turing', 'John von Neumann', 'Peter Shor',
  // cool people/friends
  'Robinlinden', 'Grayhatter', 'Zyansheep',

  // lolz
  'Doctor Who', 'Gandalf the Grey', 'Darth Vader', 'Leeroy Jenkins',
  'Sir NotAppearing Inthisfilm'
];

const CARD_TYPES = {
  VISA: {
    prefix: '4',
    length: 16
  },
  MONOPOLY: {
    prefix: 'lol',
    length: 16
  },
  MASTERCARD: {
    prefix: '51',
    length: 16
  },
  AMEX: {
    prefix: '37',
    length: 15
  },
  DISCOVER: {
    prefix: '6011',
    length: 16
  }
};

const STREET_NAMES = [
  "Binary", "Quantum", "Logic", "Silicon", "Firewall", "Bandwidth", "Protocol",
  "Algorithm", "Terabyte", "Gigabit", "Encryption", "Decryption", "Open Source",
  "Cyber", "Hacker", "Pixel", "Dank Meme", "Troll"
];

const STREET_TYPES = [
  "St.", "Ave.", "Blvd.", "Ln.", "Rd.", "Way", "Ct.", "Pl.", "Dr."
];

function get_random_int(min, max) {
  return Math.floor(Math.random() * (Math.floor(max) - Math.ceil(min) + 1)) + Math.ceil(min);
}

function calculate_checksum(number) {
  let sum = 0;
  for (let i = 0; i < number.length; i++) {
    let digit = parseInt(number[i]);

    if ((i % 2) !== length % 2) {
      sum += digit;
    } else if (digit > 4) {
      sum += 2 * digit - 9;
    } else {
      sum += 2 * digit;
    }
  }
  return (10 - (sum % 10)) % 10;
}

function generate_card_number(prefix, length) {
  let number = prefix;
  while (number.length < length - 1) {
    number += get_random_int(0, 9);
  }
  number += calculate_checksum(number);
  return number;
}

function generate_expiry_date() {
  const year = new Date().getFullYear() + get_random_int(1, 5);
  const month = String(get_random_int(1, 12)).padStart(2, '0');
  return `${month}/${year}`;
}

function generate_cvv() {
  return String(get_random_int(0, 999)).padStart(3, '0');
}

function generate_billing_address() {
  const num_street_name_parts = get_random_int(1, 3);
  let street_name = "";
  for (let i = 0; i < num_street_name_parts; i++) {
    street_name += STREET_NAMES[get_random_int(0, STREET_NAMES.length - 1)] + " ";
  }
  const street_type = STREET_TYPES[get_random_int(0, STREET_TYPES.length - 1)];
  return `${get_random_int(1, 9999)} ${street_name}${street_type}`;
}

function generate_card_details() {
  const card_type_names = Object.keys(CARD_TYPES);
  const card_type = card_type_names[get_random_int(0, card_type_names.length - 1)];
  const { prefix, length } = CARD_TYPES[card_type];
  return {
    card_type,
    card_number: generate_card_number(prefix, length),
    expiry_date: generate_expiry_date(),
    cvv: generate_cvv(),
    card_holder_name: CARD_HOLDERS[get_random_int(0, CARD_HOLDERS.length - 1)],
    billing_address: generate_billing_address()
  };
}

function display_card_details(details) {
  document.getElementById('card-details').insertAdjacentHTML('beforeend',
    "<div>\n" +
    "  <p><b>Card Holder:</b> " + details.card_holder_name + "</p>\n" +
    "  <p><b>Card Type:</b> " + details.card_type + "</p>\n" +
    "  <p><b>Card Number:</b> " + details.card_number + "</p>\n" +
    "  <p><b>Expiry Date:</b> " + details.expiry_date + "</p>\n" +
    "  <p><b>CVV:</b> " + details.cvv + "</p>\n" +
    "  <p><b>Billing Address:</b> " + details.billing_address + "</p>\n" +
    "  <hr>\n" +
    "</div>\n"
  );
}

function generate_and_display_cards(count) {
  const loading_message = document.getElementById('loading-message');
  loading_message.innerHTML = '$ Fetching cards...\t';

  const spinner = document.getElementById('spinner');

  setTimeout(() => {
    for (let i = 0; i < count; i++) {
      display_card_details(generate_card_details());
    }

    loading_message.innerHTML += 'Done!';
  }, 2000);
}

generate_and_display_cards(1);

document.getElementById('more').addEventListener('click', (event) => {
  generate_and_display_cards(100);
});

</script>

</script>
