from seahorse.prelude import *

declare_id('D3ymjhwwXbJ4eAuHrQABVVD4aGP51cZXHN7cA8eoNXfb')

@instruction
def init_model(signer: Signer):
    print("Initializing on-chain LLM model")
