/**
 * Program IDL in camelCase format in order to be used in JS/TS.
 *
 * Note that this is only a type helper and is not the actual IDL. The original
 * IDL can be found at `target/idl/sea_nn.json`.
 */
export type SeaNn = {
  "address": "CG3rq4URcAwEUNnfGZ3YHRKTZAZgyemttN1BQwypa8qj",
  "metadata": {
    "name": "seaNn",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "Created with Anchor"
  },
  "instructions": [
    {
      "name": "chat",
      "discriminator": [
        189,
        34,
        82,
        12,
        30,
        178,
        146,
        97
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "chatState",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "inputText",
          "type": "bytes"
        },
        {
          "name": "inputLength",
          "type": "u32"
        }
      ]
    },
    {
      "name": "initializeChat",
      "discriminator": [
        4,
        81,
        179,
        155,
        220,
        138,
        192,
        182
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "chat",
          "writable": true,
          "signer": true
        },
        {
          "name": "model"
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": []
    },
    {
      "name": "initializeModel",
      "discriminator": [
        150,
        196,
        232,
        118,
        138,
        43,
        140,
        244
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "registry",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "vocabSize",
          "type": "u32"
        },
        {
          "name": "embeddingDim",
          "type": "u32"
        },
        {
          "name": "hiddenDim",
          "type": "u32"
        },
        {
          "name": "contextLength",
          "type": "u32"
        },
        {
          "name": "layerCount",
          "type": "u32"
        }
      ]
    },
    {
      "name": "uploadChunk",
      "discriminator": [
        130,
        219,
        165,
        153,
        119,
        149,
        252,
        162
      ],
      "accounts": [
        {
          "name": "signer",
          "writable": true,
          "signer": true
        },
        {
          "name": "registry",
          "writable": true
        },
        {
          "name": "chunk",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "chunkIndex",
          "type": "u32"
        },
        {
          "name": "chunkType",
          "type": "u8"
        },
        {
          "name": "data",
          "type": {
            "vec": "i32"
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "chatState",
      "discriminator": [
        67,
        109,
        181,
        186,
        97,
        123,
        128,
        236
      ]
    },
    {
      "name": "modelChunk",
      "discriminator": [
        254,
        185,
        27,
        224,
        107,
        118,
        26,
        25
      ]
    },
    {
      "name": "modelRegistry",
      "discriminator": [
        174,
        72,
        180,
        46,
        185,
        165,
        246,
        200
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "e000",
      "msg": "Invalid authority"
    },
    {
      "code": 6001,
      "name": "e001",
      "msg": "Input too long"
    }
  ],
  "types": [
    {
      "name": "chatState",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "model",
            "type": "pubkey"
          },
          {
            "name": "history",
            "type": "bytes"
          },
          {
            "name": "historyLen",
            "type": "u32"
          }
        ]
      }
    },
    {
      "name": "modelChunk",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "registry",
            "type": "pubkey"
          },
          {
            "name": "chunkIndex",
            "type": "u32"
          },
          {
            "name": "chunkType",
            "type": "u8"
          },
          {
            "name": "data",
            "type": {
              "vec": "i32"
            }
          }
        ]
      }
    },
    {
      "name": "modelRegistry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "chunkCount",
            "type": "u32"
          },
          {
            "name": "vocabSize",
            "type": "u32"
          },
          {
            "name": "embeddingDim",
            "type": "u32"
          },
          {
            "name": "hiddenDim",
            "type": "u32"
          },
          {
            "name": "contextLength",
            "type": "u32"
          },
          {
            "name": "layerCount",
            "type": "u32"
          }
        ]
      }
    }
  ]
};
