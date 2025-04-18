# SEA-LMM: Bored Stacc @Web

**SEA-LMM** is a fully on-chain language model implementation built on Solana using [@seahorse_lang](https://twitter.com/seahorse_lang), following the **Bored Stacc** philosophy.

## What is Bored Stacc?

Bored Stacc is the shit I'm too lazy or too unskillful to cook. It's a minimal implementation with just enough to:
1. Demonstrate the core concept
2. Provide rough documentation (almost enough to patent it)
3. Share some saucecode/proof of concept

## How It Works

1. We post all that shit to [@RaydiumProtocol](https://twitter.com/RaydiumProtocol) launchlabs as a ticker that gets 20% creator share if it bonds
2. If it bonds, the community decides on a builder to keep building that idea to fruition
3. The original creator gets the creator share

It's [@ideasdotfun](https://twitter.com/ideasdotfun) but I can use it, now, by rethinking the meta of it!

## SEA-LMM Core Concept

SEA-LMM demonstrates a minimal viable on-chain language model:

- Fully on-chain transformer-based language model
- Trained off-chain, with weights stored on-chain
- Optimized for Solana's constraints and costs
- Inference can be performed directly from your dApp

## Directory Structure

```
sea-nn/
├── client/           # Client-side code for interacting with the model
├── contract/         # Solana program (smart contract) code
├── models/           # Pre-trained models and weights
├── data/             # Training data and examples
├── examples/         # Example implementations
└── site/             # Example frontend
```

## Technical Implementation

The on-chain model uses several optimizations to fit within Solana's constraints:

- Quantized weights and activations
- Efficient attention mechanisms
- Lookup tables for common operations
- Optimized data structures for Solana storage

## Get Started

```bash
# Clone the repo
git clone https://github.com/your-username/sea-lmm.git

# Install dependencies
cd sea-lmm
npm install

# Deploy your own instance
npm run deploy
```

## Contribute

Got skills I don't? Want to take this further? Join the community on [@RaydiumProtocol](https://twitter.com/BoredSStudio) launchlabs!

## Bored Stacc Philosophy

The web development industry moves fast - new frameworks every week, new concepts every day. Instead of chasing the new hotness, Bored Stacc embraces the "boring" approach:

> Use whatever stack you already know that can complete the interesting thing you want to build.

When you want to ship great products, learning new technologies isn't always the way. Find your Boring Stack and focus on your ideas.

Created by [@staccvoerflow](https://twitter.com/staccoverflow) and built on the original [Sea-NN](https://github.com/wireless-anon/sea-nn)