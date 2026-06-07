/*
 * BREAK SOLANA: EDITION 2
 * Design: "Glitch Manifesto" — Digital Brutalism
 * Colors: Black #000, White #FFF, Cyan #00F0FF, Red #FF2020
 * Fonts: Space Grotesk (display), JetBrains Mono (code)
 * Layout: Brutalist asymmetric, massive type, dense data
 */

import { useEffect, useRef, useState } from "react";
import { motion, useInView, useScroll, useTransform } from "framer-motion";

const HERO_IMG = "https://private-us-east-1.manuscdn.com/sessionFile/97u4t2GDpXAOmkNfG2jhgv/sandbox/7StNubBZQwK8vdyVqZalJZ-img-1_1771089333000_na1fn_aGVyby1nbGl0Y2g.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOTd1NHQyR0RwWEFPbWtOZkcyamhndi9zYW5kYm94LzdTdE51YkJaUXdLOHZkeVZxWmFsSlotaW1nLTFfMTc3MTA4OTMzMzAwMF9uYTFmbl9hR1Z5YnkxbmJHbDBZMmcucG5nP3gtb3NzLXByb2Nlc3M9aW1hZ2UvcmVzaXplLHdfMTkyMCxoXzE5MjAvZm9ybWF0LHdlYnAvcXVhbGl0eSxxXzgwIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=qAVGSCiJP0gInHPOaR6LZiqpPe0HCUzKWy6KSFl895mV5Xl0Fs-KgZpu11pSiADGzHvN~qJERhghNzs~xYmJMyzWDYE7dO69DJc~r3GB5HkCpiht-2WMRYWJlfDoYP5ob7dEfpuLjuLKENoXO6br7r8iMDbQuL8rBD1hWlB23hMZU7QolqGMRwl7a0QlFdOwmwg7aPCdvqJ~XP95eeu4Nco2zkau~D24Uv2hD32wRnQJi-bqFtqEM9v6h56sgdB~5lcqVWTr86x85IWZQCZ-igEAkz5-tOeZW1ITDw8j3Hb38MFvs10aPHrFQXX67th5bpcUcBy8ZtS7Gf6J3uPj3Q__";

const NEURAL_IMG = "https://private-us-east-1.manuscdn.com/sessionFile/97u4t2GDpXAOmkNfG2jhgv/sandbox/7StNubBZQwK8vdyVqZalJZ-img-2_1771089338000_na1fn_bmV1cmFsLW5ldHdvcms.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOTd1NHQyR0RwWEFPbWtOZkcyamhndi9zYW5kYm94LzdTdE51YkJaUXdLOHZkeVZxWmFsSlotaW1nLTJfMTc3MTA4OTMzODAwMF9uYTFmbl9ibVYxY21Gc0xXNWxkSGR2Y21zLnBuZz94LW9zcy1wcm9jZXNzPWltYWdlL3Jlc2l6ZSx3XzE5MjAsaF8xOTIwL2Zvcm1hdCx3ZWJwL3F1YWxpdHkscV84MCIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=YGNGMi68Ns0gBiZwqlz-6aJiIltZ959aeY2~ep52-caDVOEy6oaMXZ6it--o8c8Jo5cJMOMJbyMQLILGcfknZF9Zv-8UmEnacAvCKL5JyFDbbAqBSHn2g9gQoOgKtzmVc4ZiyWVkYL9QAoVmHd7CFF77NhJPo9QSYWplYJ3TDd-8rmU1zfR9Heth0SFcbFUGUCIDRCte7BMPMZuMDTLrKAkuLOJLpi56hTszXZgIMGV1amTI4VLHk2~nbmH1AHtTPP37OfAJwVpfR6w9WhyDFx9oO1l2eOiKRuC3QrjNiIL84R-zhC5rL8ye3hKznyv5eXRtlv4yd0FyyUhIq00CUA__";

const CLOCKWORK_IMG = "https://private-us-east-1.manuscdn.com/sessionFile/97u4t2GDpXAOmkNfG2jhgv/sandbox/7StNubBZQwK8vdyVqZalJZ-img-3_1771089334000_na1fn_Y2xvY2t3b3JrLWV4cGxvaXQ.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOTd1NHQyR0RwWEFPbWtOZkcyamhndi9zYW5kYm94LzdTdE51YkJaUXdLOHZkeVZxWmFsSlotaW1nLTNfMTc3MTA4OTMzNDAwMF9uYTFmbl9ZMnh2WTJ0M2IzSnJMV1Y0Y0d4dmFYUS5wbmc~eC1vc3MtcHJvY2Vzcz1pbWFnZS9yZXNpemUsd18xOTIwLGhfMTkyMC9mb3JtYXQsd2VicC9xdWFsaXR5LHFfODAiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=dIc4J25YyynydoXjX8Zy1Wd2dxD2knWruK0VjIW66KAx6cZ8fk2pfemVi0Gb8T3uHhJhG7pCDM9fV62KQyzejaDZVmbN1VYT~xbBlpPBndxnQNbYL9TgNvXXirH7X7t9k9g65f30sYt3hImeNyr6HOz5ofETWI4N0MtclZR976F0Ae3Z3aQLvRVNhsC7iH-Mh2QKYdHvwroPLL-aDRL32G906cI4WbAkHvyIpJ5l6sPP9FU7mXTMKweEqVq3ZjfTOIUwv2oj8hAtGbuQhQWs-rd7-Pbu9vMfyy~ErVujKeM~7Kusr5V8TfFEe1FU7IFs20F75iiqwWzcYsyz25CVjw__";

const BLOCK_IMG = "https://private-us-east-1.manuscdn.com/sessionFile/97u4t2GDpXAOmkNfG2jhgv/sandbox/7StNubBZQwK8vdyVqZalJZ-img-4_1771089344000_na1fn_YmxvY2stc3BhY2U.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOTd1NHQyR0RwWEFPbWtOZkcyamhndi9zYW5kYm94LzdTdE51YkJaUXdLOHZkeVZxWmFsSlotaW1nLTRfMTc3MTA4OTM0NDAwMF9uYTFmbl9ZbXh2WTJzdGMzQmhZMlUucG5nP3gtb3NzLXByb2Nlc3M9aW1hZ2UvcmVzaXplLHdfMTkyMCxoXzE5MjAvZm9ybWF0LHdlYnAvcXVhbGl0eSxxXzgwIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=QAPSBJdwG~LVwYQ9Ok4UoUlKV7FXJaMQJ~5TIEiFsPjCryQcv0pRd5Iv1cZgOk6uxOuNoS8FxKmkx5XW6hIoPAv91aiywcq0AH6WUYnLWfgZR4CC3AwjgWBoQqAOigXBpMGsX7-mAtTsoUcr-UGpMjzxrHfU4DDBQCshdTchVmVB2sbYogZm7FCrKOZylUYMpdE6Bourfq~bAjXRNyyjQjK0rJzXyuO-g541FKb00zHbvm9r5Uz4aJ2tmxCSDuEthyoeUpqlPjIMj-MxpmSYzlGpQxoIy8q2hdyAfmtkM4awYVB6QNTWjC0V1pVwOt38JFYpnTRxJSDNk5jfLyllXQ__";

const MAGIC_WORD_IMG = "https://private-us-east-1.manuscdn.com/sessionFile/97u4t2GDpXAOmkNfG2jhgv/sandbox/gL0sNSiDVLdemM8ICyju1N-img-1_1771105320000_na1fn_bWFnaWMtd29yZC1oZXJv.png?x-oss-process=image/resize,w_1920,h_1920/format,webp/quality,q_80&Expires=1798761600&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOTd1NHQyR0RwWEFPbWtOZkcyamhndi9zYW5kYm94L2dMMHNOU2lEVkxkZW1NOElDeWp1MU4taW1nLTFfMTc3MTEwNTMyMDAwMF9uYTFmbl9iV0ZuYVdNdGQyOXlaQzFvWlhKdi5wbmc~eC1vc3MtcHJvY2Vzcz1pbWFnZS9yZXNpemUsd18xOTIwLGhfMTkyMC9mb3JtYXQsd2VicC9xdWFsaXR5LHFfODAiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=htd3CYDwHYKsAXY6NkFdlB3Hwbo98t5L-qoHgbtkoyWw2WHaPyWNl6lMY0na~UNn8FKSgUV48r6jS~HDitT~iv2hg6nreZtIeAedT0IcKMMKl1OPKaRIVvepUSn90rUOd95LiYXPTK9yGXhrLIVFJ1n~WTKXYENncvg1kPG~g7fp6TZfZRaCYlP6h1ZfuLDOthhSZVuEIJJLM3rwun8JxVjpUzNlxV58Ljr5Xe6QSFH1E8g5a2XUCt6GVe-IAuWZFBWQ1RGl4gNiMgYs8W~NJ-741WNoKSnIVxCUmbUQvAwarHiIyADd4YJ3SyzoZPffk5df532zi8g1cz5TBTYcCQ__";

// ─── Program Constants ────────────────────────────────────────
const PROGRAM_ID = "Dwu4RUjRPhYkvmwVaY6NdRZMyN5yKuxWR3Vi1SQrwEma";
const POT_PDA = "AvP6URQbeEwuEXkf97gqGkb5k6RvdpFWRSBcGou7muZR";
const WALLET_ADDRESS = "89VB5UmvopuCFmp5Mf8YPX28fGvvqn79afCgouQuPyhY";
const BANNED_TOKENS = [5536, 6139, 10883, 22294, 22975, 32707, 34850, 42055, 44546];
const BANNED_WORDS = ['" magic"', '" Magic"', '" magical"', '" Magical"', '"Magic"', '"magic"', '" magically"', '" magician"', '"Magicka"'];

const HELIUS_API_KEY = "YOUR_HELIUS_API_KEY";
const HELIUS_RPC = `https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`;
const HELIUS_WS = `wss://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`;
const VOCAB_CDN = "/vocab.json";

// State account layout offsets
const STATE_HEADER_SIZE = 16;
const STATE_TOKEN_AREA = STATE_HEADER_SIZE; // offset 16
const STATE_SEQ_LEN_OFF = 3;
const STATE_GEN_COUNT_OFF = 5;
const STATE_PRED_TOKEN_OFF = 8;
const STATE_HAS_WON_OFF = 12;

function SolGoalMeter() {
  const [balance, setBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [live, setLive] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true });

  // Initial HTTP fetch + fallback polling
  useEffect(() => {
    const RPC_ENDPOINTS = [
      HELIUS_RPC,
      "https://solana-rpc.publicnode.com",
      "https://api.mainnet-beta.solana.com",
    ];
    async function fetchBalance() {
      for (const rpc of RPC_ENDPOINTS) {
        try {
          const res = await fetch(rpc, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: 1,
              method: "getBalance",
              params: [POT_PDA],
            }),
          });
          const data = await res.json();
          if (data.result?.value !== undefined) {
            setBalance(data.result.value / 1e9);
            setError(false);
            setLoading(false);
            return;
          }
        } catch {
          continue;
        }
      }
      setError(true);
      setLoading(false);
    }
    fetchBalance();
    // Fallback polling every 30s in case WS disconnects
    const interval = setInterval(fetchBalance, 30000);
    return () => clearInterval(interval);
  }, []);

  // WebSocket subscription for real-time updates
  useEffect(() => {
    let ws: WebSocket | null = null;
    let subId: number | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      try {
        ws = new WebSocket(HELIUS_WS);

        ws.onopen = () => {
          setLive(true);
          // Subscribe to account changes
          ws?.send(JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "accountSubscribe",
            params: [
              POT_PDA,
              { encoding: "jsonParsed", commitment: "confirmed" },
            ],
          }));
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            // Subscription confirmation
            if (msg.id === 1 && msg.result !== undefined) {
              subId = msg.result;
            }
            // Account change notification
            if (msg.method === "accountNotification" && msg.params?.result?.value) {
              const lamports = msg.params.result.value.lamports;
              if (typeof lamports === "number") {
                setBalance(lamports / 1e9);
                setError(false);
                setLoading(false);
              }
            }
          } catch { /* ignore parse errors */ }
        };

        ws.onclose = () => {
          setLive(false);
          // Reconnect after 5 seconds
          reconnectTimer = setTimeout(connect, 5000);
        };

        ws.onerror = () => {
          setLive(false);
          ws?.close();
        };
      } catch {
        setLive(false);
        reconnectTimer = setTimeout(connect, 5000);
      }
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws && subId !== null) {
        try {
          ws.send(JSON.stringify({
            jsonrpc: "2.0",
            id: 2,
            method: "accountUnsubscribe",
            params: [subId],
          }));
        } catch { /* ignore */ }
      }
      ws?.close();
    };
  }, []);

  const displayBalance = balance !== null ? balance.toFixed(2) : "—";

  return (
    <div ref={ref} className="border border-[#00F0FF]/20 bg-black p-6 md:p-8 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <p className="font-mono text-[#00F0FF] text-xs tracking-widest flex items-center gap-2">
          // PRIZE POT — LIVE ON MAINNET
          {live && (
            <span className="flex items-center gap-1">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
              </span>
              <span className="text-green-400 text-[10px]">LIVE</span>
            </span>
          )}
        </p>
        <a
          href={`https://solscan.io/account/${POT_PDA}`}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-white/50 text-xs hover:text-[#00F0FF] transition-colors"
        >
          VERIFY ON-CHAIN →
        </a>
      </div>

      {/* Pot balance display */}
      <div className="relative h-16 border border-[#00F0FF]/30 bg-[#00F0FF]/[0.03] overflow-hidden mb-4 flex items-center justify-center">
        <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(0,0,0,0.15)_2px,rgba(0,0,0,0.15)_4px)]" />
        <div className="relative z-10 text-center">
          <span className="font-mono text-3xl md:text-4xl font-bold text-[#00F0FF]">
            {loading ? "LOADING..." : error ? "RPC ERROR" : `${displayBalance} SOL`}
          </span>
        </div>
      </div>

      {/* Pot PDA address */}
      <div className="flex items-center justify-between font-mono text-xs">
        <span className="text-white/50">POT PDA: <span className="text-white/70">{POT_PDA.slice(0, 16)}...{POT_PDA.slice(-8)}</span></span>
        <span className="text-white/50">GROWS WITH EVERY PLAY</span>
      </div>
    </div>
  );
}

// ─── Live Activity Feed ────────────────────────────────────
interface PlayEntry {
  signature: string;
  time: number;
  user: string;
  potContribution: string;
  devFee: string;
  seq: number;
  status: "playing" | "lost" | "won";
  promptTokens: number[];
  promptText: string;
  outputText: string;
  stateAccount: string;
}

// Vocab cache — loaded once from CDN
let vocabCache: Record<string, string> | null = null;
async function loadVocab(): Promise<Record<string, string>> {
  if (vocabCache) return vocabCache;
  try {
    const res = await fetch(VOCAB_CDN);
    vocabCache = await res.json();
    return vocabCache!;
  } catch {
    return {};
  }
}

// Decode GPT-2 token IDs to readable text
function decodeTokens(ids: number[], vocab: Record<string, string>): string {
  return ids.map(id => {
    const raw = vocab[String(id)] || `[${id}]`;
    // GPT-2 uses Ġ for leading space
    return raw.replace(/Ġ/g, " ").replace(/Ċ/g, "\n");
  }).join("");
}

// Parse u16 LE from base64-decoded bytes
function readU16LE(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8);
}

function LiveActivityFeed() {
  const [plays, setPlays] = useState<PlayEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [totalPlays, setTotalPlays] = useState(0);
  const [expandedSig, setExpandedSig] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {

    async function fetchRecentPlays() {
      try {
        const vocab = await loadVocab();

        // Get recent signatures for the pot PDA
        const res = await fetch(HELIUS_RPC, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            jsonrpc: "2.0", id: 1,
            method: "getSignaturesForAddress",
            params: [POT_PDA, { limit: 50 }],
          }),
        });
        const data = await res.json();
        const sigs = (data.result || []).filter((s: { err: unknown }) => !s.err);

        // Fetch all transactions in parallel (batch of 5)
        const txResults: { sig: string; blockTime: number; tx: any }[] = [];
        for (let i = 0; i < sigs.length; i += 5) {
          const batch = sigs.slice(i, i + 5);
          const results = await Promise.all(
            batch.map(async (sig: { signature: string; blockTime: number }) => {
              try {
                const txRes = await fetch(HELIUS_RPC, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    jsonrpc: "2.0", id: 1,
                    method: "getTransaction",
                    params: [sig.signature, { maxSupportedTransactionVersion: 0, encoding: "json" }],
                  }),
                });
                const txData = await txRes.json();
                return { sig: sig.signature, blockTime: sig.blockTime, tx: txData.result };
              } catch { return null; }
            })
          );
          txResults.push(...results.filter(Boolean) as typeof txResults);
        }

        // Categorize transactions: INIT (has Game: log), CLOSE (has Session closed log), INIT_POT (has Pot initialized)
        type InitPlay = {
          sig: string; time: number; user: string; pot: string; dev: string; seq: number;
          promptTokens: number[]; promptText: string; stateAccount: string;
        };
        type CloseInfo = { sig: string; user: string; time: number; output: string; won: boolean };

        const initPlays: InitPlay[] = [];
        const closeInfos: CloseInfo[] = [];

        for (const { sig, blockTime, tx } of txResults) {
          if (!tx) continue;
          const logs: string[] = tx.meta?.logMessages || [];
          const accountKeys = tx.transaction?.message?.accountKeys || [];
          const ixs = tx.transaction?.message?.instructions || [];

          const hasGameLog = logs.some((l: string) => l.includes("Game:"));
          const hasCloseLog = logs.some((l: string) => l.includes("Session closed"));
          const hasPotInit = logs.some((l: string) => l.includes("Pot initialized"));

          if (hasPotInit && !hasGameLog && !hasCloseLog) continue; // skip init_pot

          if (hasGameLog) {
            // This is an INIT transaction
            const gameLog = logs.find((l: string) => l.includes("Game:")) || "";
            const userMatch = gameLog.match(/user=([A-Za-z0-9]+)/);
            const potMatch = gameLog.match(/pot=([0-9.]+)SOL/);
            const devMatch = gameLog.match(/dev=([0-9.]+)SOL/);
            const seqMatch = gameLog.match(/seq=(\d+)/);
            if (!userMatch) continue;

            let promptTokens: number[] = [];
            let promptText = "";
            let stateAccount = "";
            try {
              for (const ix of ixs) {
                const progKey = accountKeys[ix.programIdIndex];
                if (progKey === PROGRAM_ID && ix.data) {
                  const raw = decodeBase58(ix.data);
                  if (raw.length > 0 && raw[0] === 2) {
                    const seqLen = readU16LE(raw, 1);
                    if (seqLen > 0 && seqLen <= 32 && raw.length >= 3 + seqLen * 2) {
                      for (let t = 0; t < seqLen; t++) {
                        promptTokens.push(readU16LE(raw, 3 + t * 2));
                      }
                      promptText = decodeTokens(promptTokens, vocab).trim();
                    }
                  }
                  if (ix.accounts?.length > 0) stateAccount = accountKeys[ix.accounts[0]];
                }
              }
            } catch { /* ignore */ }

            initPlays.push({
              sig, time: blockTime, user: userMatch[1],
              pot: potMatch ? potMatch[1] : "0.8",
              dev: devMatch ? devMatch[1] : "0.2",
              seq: seqMatch ? parseInt(seqMatch[1]) : 0,
              promptTokens, promptText, stateAccount,
            });
          }

          if (hasCloseLog) {
            // This is a CLOSE transaction — extract output from logs
            const closerLog = logs.find((l: string) => l.includes("Session closed")) || "";
            const userMatch = closerLog.match(/user=([A-Za-z0-9]+)/) ||
              // Fallback: get user from account keys (payer is usually index 0)
              (accountKeys.length > 0 ? [null, accountKeys[0]] : null);
            const won = logs.some((l: string) => l.includes("WIN") || l.includes("WINNER"));

            // Try to extract output tokens from close logs
            let output = "";
            const outputLog = logs.find((l: string) => l.includes("output=") || l.includes("tokens="));
            if (outputLog) {
              const m = outputLog.match(/output=(.+)/);
              if (m) output = m[1];
            }

            closeInfos.push({
              sig, user: userMatch ? userMatch[1] : "", time: blockTime, output, won,
            });
          }
        }

        // Build final entries: match INIT plays with CLOSE info by user
        const entries: PlayEntry[] = [];
        for (const init of initPlays) {
          // Find a matching close for this user that happened AFTER the init
          const close = closeInfos.find(c => c.user === init.user && c.time >= init.time);
          let status: "playing" | "lost" | "won" = "playing";
          let outputText = "";

          if (close) {
            status = close.won ? "won" : "lost";
            outputText = close.output;
          }

          // If still playing or close had no output, try reading state account
          if (!outputText && init.stateAccount) {
            try {
              const stateRes = await fetch(HELIUS_RPC, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  jsonrpc: "2.0", id: 1,
                  method: "getAccountInfo",
                  params: [init.stateAccount, { encoding: "base64" }],
                }),
              });
              const stateData = await stateRes.json();
              const acctData = stateData.result?.value?.data;
              if (acctData && acctData[0]) {
                const bytes = Uint8Array.from(atob(acctData[0]), c => c.charCodeAt(0));
                if (bytes.length > STATE_TOKEN_AREA) {
                  const seqLen = readU16LE(bytes, STATE_SEQ_LEN_OFF);
                  const genCount = readU16LE(bytes, STATE_GEN_COUNT_OFF);
                  const predToken = readU16LE(bytes, STATE_PRED_TOKEN_OFF);
                  const genTokens: number[] = [];
                  for (let g = 0; g < genCount && (seqLen + g) < 32; g++) {
                    genTokens.push(readU16LE(bytes, STATE_TOKEN_AREA + (seqLen + g) * 2));
                  }
                  if (genCount > 0 && predToken > 0 && (genTokens.length === 0 || genTokens[genTokens.length - 1] !== predToken)) {
                    genTokens.push(predToken);
                  }
                  if (genTokens.length > 0) outputText = decodeTokens(genTokens, vocab);
                }
              } else if (status === "playing") {
                // State account doesn't exist anymore = session was closed
                status = "lost";
              }
            } catch { /* ignore */ }
          }

          entries.push({
            signature: init.sig,
            time: init.time,
            user: init.user,
            potContribution: init.pot,
            devFee: init.dev,
            seq: init.seq,
            status,
            promptTokens: init.promptTokens,
            promptText: init.promptText,
            outputText,
            stateAccount: init.stateAccount,
          });
        }

        // Sort by time descending (newest first)
        entries.sort((a, b) => b.time - a.time);
        setPlays(entries);
        setTotalPlays(entries.length);
        setLoading(false);
      } catch (err) {
        console.error("Feed error:", err);
        setLoading(false);
      }
    }

    fetchRecentPlays();
    const interval = setInterval(fetchRecentPlays, 30000);
    return () => clearInterval(interval);
  }, []);

  function timeAgo(ts: number) {
    const diff = Math.floor(Date.now() / 1000) - ts;
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  }

  function shortAddr(addr: string) {
    return addr.slice(0, 4) + ".." + addr.slice(-4);
  }

  return (
    <div ref={ref}>
      <Reveal>
        <p className="font-mono text-[#00F0FF] text-xs tracking-widest mb-4 flex items-center gap-2">
          // LIVE ACTIVITY FEED
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
          </span>
        </p>
      </Reveal>

      <Reveal delay={0.1}>
        <div className="border border-white/10 bg-white/[0.02] overflow-hidden">
          {/* Header */}
          <div className="grid grid-cols-12 gap-2 px-4 py-3 border-b border-white/10 font-mono text-[10px] tracking-widest text-white/50">
            <span className="col-span-2">PLAYER</span>
            <span className="col-span-2">TIME</span>
            <span className="col-span-3">PROMPT</span>
            <span className="col-span-3">OUTPUT</span>
            <span className="col-span-2 text-right">STATUS</span>
          </div>

          {loading ? (
            <div className="px-4 py-8 text-center">
              <span className="font-mono text-white/50 text-sm animate-pulse">LOADING VOCAB & SCANNING CHAIN...</span>
            </div>
          ) : plays.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <span className="font-mono text-white/50 text-sm">NO PLAYS YET — BE THE FIRST</span>
            </div>
          ) : (
            <div className="max-h-[480px] overflow-y-auto">
              {plays.map((play, i) => (
                <div key={play.signature + i}>
                  {/* Main row */}
                  <div
                    onClick={() => setExpandedSig(expandedSig === play.signature ? null : play.signature)}
                    className="grid grid-cols-12 gap-2 px-4 py-3 border-b border-white/5 font-mono text-xs hover:bg-white/[0.03] transition-colors cursor-pointer group"
                  >
                    <span className="col-span-2 text-white/60 group-hover:text-[#00F0FF] transition-colors">
                      {shortAddr(play.user)}
                    </span>
                    <span className="col-span-2 text-white/50">
                      {timeAgo(play.time)}
                    </span>
                    <span className="col-span-3 text-white/70 truncate" title={play.promptText}>
                      {play.promptText ? `"${play.promptText}"` : "—"}
                    </span>
                    <span className={`col-span-3 truncate ${
                      play.outputText?.includes(" magic") ? "text-[#FFD700] font-bold" : "text-[#00F0FF]/60"
                    }`} title={play.outputText}>
                      {play.outputText ? play.outputText : play.status === "playing" ? (
                        <span className="animate-pulse">generating...</span>
                      ) : (
                        <span className="text-white/40 italic">session closed</span>
                      )}
                    </span>
                    <span className={`col-span-2 text-right font-bold ${
                      play.status === "won" ? "text-[#FFD700]" :
                      play.status === "lost" ? "text-[#FF2020]/50" :
                      "text-[#00F0FF] animate-pulse"
                    }`}>
                      {play.status === "won" ? "WON" :
                       play.status === "lost" ? "LOST" :
                       "PLAYING..."}
                    </span>
                  </div>

                  {/* Expanded detail row */}
                  {expandedSig === play.signature && (
                    <div className="px-4 py-4 border-b border-white/5 bg-white/[0.01] space-y-3">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="font-mono text-[10px] text-white/50 tracking-widest mb-1">PROMPT</p>
                          <p className="font-mono text-sm text-white/90">
                            "{play.promptText || "unknown"}"
                          </p>
                          {play.promptTokens.length > 0 && (
                            <p className="font-mono text-[10px] text-white/40 mt-1">
                              TOKENS: [{play.promptTokens.join(", ")}]
                            </p>
                          )}
                        </div>
                        <div>
                          <p className="font-mono text-[10px] text-white/50 tracking-widest mb-1">OUTPUT</p>
                          <p className={`font-mono text-sm ${
                            play.outputText?.includes(" magic") ? "text-[#FFD700]" : "text-[#00F0FF]/80"
                          }`}>
                            {play.outputText || (play.status === "playing" ? "Still generating..." : "Output not available — state account closed")}
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-4 font-mono text-[10px] text-white/40">
                        <span>POT: +{play.potContribution} SOL</span>
                        <span>DEV: {play.devFee} SOL</span>
                        <span>PLAY #{play.seq}</span>
                        {play.stateAccount && (
                          <a
                            href={`https://solscan.io/account/${play.stateAccount}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[#00F0FF]/40 hover:text-[#00F0FF] transition-colors"
                            onClick={e => e.stopPropagation()}
                          >
                            STATE: {play.stateAccount.slice(0, 8)}...
                          </a>
                        )}
                        <a
                          href={`https://solscan.io/tx/${play.signature}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[#00F0FF]/40 hover:text-[#00F0FF] transition-colors"
                          onClick={e => e.stopPropagation()}
                        >
                          TX: {play.signature.slice(0, 8)}...
                        </a>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Footer */}
          <div className="px-4 py-3 border-t border-white/10 flex justify-between font-mono text-[10px] text-white/40">
            <span>TOTAL PLAYS: {totalPlays}</span>
            <span>CLICK ROW TO EXPAND · AUTO-REFRESHES EVERY 30S</span>
          </div>
        </div>
      </Reveal>
    </div>
  );
}

// Base58 decoder for Solana instruction data
const BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
function decodeBase58(str: string): Uint8Array {
  const bytes: number[] = [0];
  for (const char of str) {
    const idx = BASE58_ALPHABET.indexOf(char);
    if (idx < 0) return new Uint8Array(0);
    let carry = idx;
    for (let j = 0; j < bytes.length; j++) {
      carry += bytes[j] * 58;
      bytes[j] = carry & 0xff;
      carry >>= 8;
    }
    while (carry > 0) {
      bytes.push(carry & 0xff);
      carry >>= 8;
    }
  }
  // Leading zeros
  for (const char of str) {
    if (char !== "1") break;
    bytes.push(0);
  }
  return new Uint8Array(bytes.reverse());
}

// ─── Animated Counter ───────────────────────────────────────
function Counter({ end, suffix = "", prefix = "", duration = 2 }: { end: number; suffix?: string; prefix?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const step = end / (duration * 60);
    const timer = setInterval(() => {
      start += step;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 1000 / 60);
    return () => clearInterval(timer);
  }, [inView, end, duration]);

  return (
    <span ref={ref} className="tabular-nums">
      {prefix}{typeof end === 'number' && end % 1 !== 0 ? count.toFixed(1) : count.toLocaleString()}{suffix}
    </span>
  );
}

// ─── Typing Effect ──────────────────────────────────────────
function TypingText({ text, className = "" }: { text: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (!inView) return;
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 50);
    return () => clearInterval(timer);
  }, [inView, text]);

  return (
    <span ref={ref} className={className}>
      {displayed}
      <span className="inline-block w-[2px] h-[1em] bg-[#00F0FF] ml-1 animate-pulse" />
    </span>
  );
}

// ─── Glitch Heading ─────────────────────────────────────────
function GlitchHeading({ children, className = "" }: { children: string; className?: string }) {
  return (
    <h2
      className={`glitch-text font-bold tracking-tighter ${className}`}
      data-text={children}
    >
      {children}
    </h2>
  );
}

// ─── Section Reveal ─────────────────────────────────────────
function Reveal({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// ─── MAIN PAGE ──────────────────────────────────────────────
export default function Home() {
  const { scrollYProgress } = useScroll();
  const heroOpacity = useTransform(scrollYProgress, [0, 0.15], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 0.15], [1, 0.95]);

  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      {/* Scanline overlay */}
      <div className="scanline-overlay" />

      {/* ═══ HERO ═══════════════════════════════════════════ */}
      <motion.section
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-screen flex flex-col justify-end pb-12 md:pb-20"
      >
        {/* Background */}
        <div className="absolute inset-0 z-0">
          <img
            src={HERO_IMG}
            alt=""
            className="w-full h-full object-cover opacity-40"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black via-black/60 to-transparent" />
        </div>

        {/* Top bar */}
        <div className="relative z-10 w-full border-b border-white/10 px-4 md:px-8 py-3 flex justify-between items-center font-mono text-xs text-white/50">
          <span>STACCOVERFLOW // 2026</span>
          <span className="text-[#00F0FF]">[ EDITION 2 ]</span>
          <span>PROGRAM DEPLOYED: <span className="text-[#00F0FF]">MAINNET</span></span>
        </div>

        {/* Hero content */}
        <div className="relative z-10 px-4 md:px-8 lg:px-16 mt-auto">
          <motion.div
            initial={{ opacity: 0, x: -80 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <p className="font-mono text-[#00F0FF] text-sm md:text-base mb-4 tracking-widest uppercase">
              &gt; deploying neural network to blockchain_
            </p>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 60 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, delay: 0.4 }}
            className="text-[clamp(3rem,12vw,11rem)] font-bold leading-[0.85] tracking-tighter text-white"
          >
            BREAK<br />
            <span className="text-[#00F0FF] glitch-text" data-text="SOLANA">SOLANA</span>
          </motion.h1>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.0 }}
            className="mt-6 flex items-center gap-6"
          >
            <span className="inline-block bg-[#FF2020] text-black font-bold text-sm md:text-base px-4 py-2 tracking-wider">
              EDITION 2
            </span>
            <span className="font-mono text-white/50 text-sm">
              AN LLM THAT EATS 25% OF THE NETWORK — NOW A GAME
            </span>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 1.3 }}
            className="mt-8 max-w-2xl text-white/80 text-base md:text-lg leading-relaxed"
          >
            We deployed a full GPT neural network as a Solana program on mainnet. Every matrix multiplication,
            every attention head, every token — computed entirely on-chain. One sentence consumes
            25% of Solana's total compute capacity for a full minute. Program live at{' '}
            <a href={`https://solscan.io/account/${PROGRAM_ID}`} target="_blank" rel="noopener noreferrer" className="text-[#00F0FF] underline">{PROGRAM_ID.slice(0,8)}...{PROGRAM_ID.slice(-4)}</a>.
          </motion.p>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10"
        >
          <div className="w-[1px] h-12 bg-gradient-to-b from-[#00F0FF] to-transparent animate-pulse" />
        </motion.div>
      </motion.section>

      {/* ═══ POT + LIVE FEED (right after hero) ═══════════════ */}
      <section className="relative border-t border-[#FF2020]/30 py-8 md:py-12 bg-black">
        <div className="px-4 md:px-8 lg:px-16">
          {/* Pot balance strip */}
          <div className="flex flex-col md:flex-row items-start md:items-center gap-6 md:gap-12 mb-8">
            <div className="flex-1">
              <SolGoalMeter />
            </div>
            <div className="flex items-center gap-4">
              <a
                href="https://solscan.io/account/AvP6URQbeEwuEXkf97gqGkb5k6RvdpFWRSBcGou7muZR"
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-xs text-[#00F0FF]/80 hover:text-[#00F0FF] transition-colors"
              >
                VERIFY POT ON SOLSCAN →
              </a>
            </div>
          </div>

          {/* Live feed */}
          <LiveActivityFeed />
        </div>
      </section>

      {/* ═══ SAY THE MAGIC WORD — GAME ═══════════════════════ */}
      <section className="relative border-t border-[#FF2020]/30 py-20 md:py-32 overflow-hidden">
        {/* Background image */}
        <div className="absolute inset-0 z-0">
          <img src={MAGIC_WORD_IMG} alt="" className="w-full h-full object-cover opacity-20" />
          <div className="absolute inset-0 bg-gradient-to-b from-black via-black/80 to-black" />
        </div>

        <div className="relative z-10 px-4 md:px-8 lg:px-16">
          <Reveal>
            <div className="text-center mb-12">
              <p className="font-mono text-[#FF2020] text-xs tracking-widest mb-4 animate-pulse">
                // NEW: ON-CHAIN GAME — LIVE NOW
              </p>
              <h2 className="text-5xl md:text-8xl lg:text-[10rem] font-bold tracking-tighter leading-[0.85] mb-6">
                SAY THE<br />
                <span className="text-[#FF2020] glitch-text" data-text="MAGIC WORD">MAGIC WORD</span>
              </h2>
              <p className="text-white/80 text-lg md:text-xl max-w-3xl mx-auto">
                Pay 1 SOL. Choose a prompt. Run on-chain inference. If the LLM generates the token
                <span className="text-[#FF2020] font-bold"> "magic"</span> — you win the entire pot.
                But you can't cheat — <span className="text-[#FF2020]">all magic-related tokens are banned from prompts</span>.
              </p>
            </div>
          </Reveal>

          {/* Game Rules Grid */}
          <Reveal delay={0.2}>
            <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-[1px] bg-white/10 max-w-5xl mx-auto">
              {[
                { step: "01", title: "PAY TO PLAY", detail: "1 SOL entry fee — 0.8 SOL goes to the prize pot, 0.2 SOL dev fee (apes & burns token). Plus ~1 SOL refundable account rent." },
                { step: "02", title: "CHOOSE YOUR PROMPT", detail: "Pick up to 32 tokens of input. No magic-related words allowed — the program rejects all 9 banned tokens on-chain." },
                { step: "03", title: "WIN THE POT", detail: "If ANY generated token is ' magic' (token #5536 in GPT-2 vocab), you win the ENTIRE prize pot instantly." },
              ].map((item, i) => (
                <div key={i} className="bg-black p-8 md:p-10 group hover:bg-white/[0.03] transition-colors duration-500">
                  <p className="font-mono text-[#FF2020] text-4xl font-bold mb-4">{item.step}</p>
                  <p className="text-white font-bold text-lg mb-3">{item.title}</p>
                  <p className="text-white/60 text-sm leading-relaxed">{item.detail}</p>
                </div>
              ))}
            </div>
          </Reveal>

          {/* How it works detail */}
          <Reveal delay={0.3}>
            <div className="mt-8 border border-[#00F0FF]/20 bg-black/80 p-6 md:p-8 max-w-5xl mx-auto">
              <p className="font-mono text-[#00F0FF] text-xs tracking-widest mb-4">// HOW THE GAME WORKS ON-CHAIN</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <p className="text-white/80 text-sm leading-relaxed mb-4">
                    The game runs entirely on Solana. Your entry fee is split: <span className="text-[#00F0FF] font-bold">80% to the prize pot PDA</span> and
                    <span className="text-[#FF2020] font-bold"> 20% dev fee</span> (used to buy & burn the token on Axiom). The pot grows with every player until someone wins.
                  </p>
                  <p className="text-white/80 text-sm leading-relaxed mb-4">
                    When you close your session, the ~1 SOL account rent also goes to the pot — so the pot grows
                    even from losing plays. The magic token is <span className="font-mono text-[#FF2020]">" magic"</span> (token ID 5536 in GPT-2's vocabulary).
                  </p>
                  <p className="text-white/80 text-sm leading-relaxed">
                    <span className="text-[#FF2020] font-bold">Anti-cheat:</span> The Rust program rejects prompts containing any of 9 magic-related tokens:
                    {BANNED_WORDS.map((w, i) => <span key={i} className="font-mono text-[#FF2020]/70 text-xs">{i > 0 ? ', ' : ''}{w}</span>)}.
                    You must trick the model into saying it without prompting it directly.
                  </p>
                </div>
                <div className="border border-white/10 bg-white/[0.02] p-4 font-mono text-xs">
                  <p className="text-white/50 mb-2">// Fee breakdown per play</p>
                  <p><span className="text-[#00F0FF]">ENTRY_FEE</span>    = 1.0 SOL</p>
                  <p><span className="text-[#00F0FF]">  → POT</span>      = 0.8 SOL <span className="text-white/50">(80%)</span></p>
                  <p><span className="text-[#FF2020]">  → DEV</span>      = 0.2 SOL <span className="text-white/50">(20%)</span></p>
                  <p className="mt-2"><span className="text-white/60">RENT_DEPOSIT</span> = ~1.0 SOL <span className="text-white/50">(refundable → pot on close)</span></p>
                  <p className="mt-2 border-t border-white/10 pt-2"><span className="text-[#FF2020]">MAGIC_TOKEN</span>  = 5536 <span className="text-white/50">(" magic")</span></p>
                  <p><span className="text-[#00F0FF]">WIN</span>          = ENTIRE POT</p>
                  <p className="mt-2 border-t border-white/10 pt-2"><span className="text-white/60">PROGRAM</span>    = <a href={`https://solscan.io/account/${PROGRAM_ID}`} target="_blank" rel="noopener noreferrer" className="text-[#00F0FF] hover:underline">{PROGRAM_ID.slice(0,12)}...</a></p>
                  <p><span className="text-white/60">POT_PDA</span>    = <a href={`https://solscan.io/account/${POT_PDA}`} target="_blank" rel="noopener noreferrer" className="text-[#00F0FF] hover:underline">{POT_PDA.slice(0,12)}...</a></p>
                  <p><span className="text-white/60">DEV_FEE</span>    = <span className="text-[#FF2020]">BUYS & BURNS TOKEN</span></p>
                </div>
              </div>
            </div>
          </Reveal>

          {/* Install CTA */}
          <Reveal delay={0.35}>
            <div className="mt-8 max-w-5xl mx-auto">
              <div className="border border-[#FF2020]/30 bg-[#FF2020]/5 p-6 md:p-8">
                <div className="flex flex-col md:flex-row items-center gap-6">
                  <div className="flex-1">
                    <p className="font-mono text-[#FF2020] text-xs tracking-widest mb-3">// INSTALL & PLAY</p>
                    <div className="bg-black border border-white/10 p-4 font-mono text-sm mb-3">
                      <p className="text-white/50 mb-1">$ npm install -g solana-llm</p>
                      <p className="text-[#00F0FF]">$ solana-llm-game -c config.json "Tell me a story"</p>
                    </div>
                    <p className="text-white/60 text-xs mb-3">
                      Published on npm as <span className="text-[#00F0FF]">solana-llm@1.1.0</span> — includes both the inference CLI and game client.
                    </p>
                    <div className="bg-black border border-white/10 p-4 font-mono text-xs">
                      <p className="text-white/50 mb-2">// Quick start guide</p>
                      <p className="text-white/70">1. Install Node.js 18+</p>
                      <p className="text-[#00F0FF]">   $ npm install -g solana-llm</p>
                      <p className="text-white/70 mt-2">2. Create a config.json with your RPC + keypair</p>
                      <p className="text-[#00F0FF]">   {'{'} "rpc": "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY",</p>
                      <p className="text-[#00F0FF]">     "keypair": "/path/to/wallet.json",</p>
                      <p className="text-[#00F0FF]">     "program_id": "{PROGRAM_ID}" {'}'}</p>
                      <p className="text-white/70 mt-2">3. Check the pot & play</p>
                      <p className="text-[#00F0FF]">   $ solana-llm-game -c config.json --pot-info</p>
                      <p className="text-[#00F0FF]">   $ solana-llm-game -c config.json "Tell me a story"</p>
                    </div>
                  </div>
                  <div className="flex flex-col gap-3 shrink-0">
                    <a
                      href="https://www.npmjs.com/package/solana-llm"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-2 bg-[#FF2020] text-white font-bold px-8 py-4 text-lg hover:bg-[#FF4040] transition-colors whitespace-nowrap animate-pulse"
                    >
                      VIEW ON NPM →
                    </a>
                    <a
                      href="https://axiom.trade/@raycc"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center gap-2 border border-white/20 text-white font-bold px-8 py-3 text-sm hover:border-[#00F0FF] hover:text-[#00F0FF] transition-colors whitespace-nowrap"
                    >
                      BUY TOKEN ON AXIOM
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* Live feed moved to after hero */}

      {/* ═══ THE OUTPUT ═════════════════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16">
          <Reveal>
            <p className="font-mono text-[#00F0FF] text-xs tracking-widest mb-8">
              // ON-CHAIN OUTPUT — VERIFIED ON SOLANA MAINNET
            </p>
          </Reveal>

          <Reveal delay={0.2}>
            <div className="border border-white/10 bg-white/[0.02] p-6 md:p-10 max-w-4xl">
              <p className="font-mono text-white/50 text-xs mb-4">PROMPT:</p>
              <p className="text-2xl md:text-4xl font-bold text-white/60 mb-6">
                "Once upon a time"
              </p>
              <p className="font-mono text-white/50 text-xs mb-4">OUTPUT:</p>
              <p className="text-2xl md:text-4xl font-bold">
                <span className="text-white/60">"Once upon a time</span>
                <TypingText
                  text=", there was a little girl named Lily."
                  className="text-[#00F0FF]"
                />
                <span className="text-white/60">"</span>
              </p>
              <div className="mt-8 flex flex-wrap gap-4 font-mono text-xs text-white/50">
                <span>TOKENS: 11, 612, 373, 257, 1310, 2576, 3706, 20037, 13</span>
                <span className="text-white/30">|</span>
                <span>TXS: 2,083</span>
                <span className="text-white/30">|</span>
                <span>WON: NO</span>
                <span className="text-white/30">|</span>
                <span className="text-[#00F0FF]">MATCHES HUGGINGFACE REFERENCE ✓</span>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ THE NUMBERS ═══════════════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32 noise-bg">
        <div className="px-4 md:px-8 lg:px-16">
          <Reveal>
            <GlitchHeading className="text-5xl md:text-8xl lg:text-[10rem] mb-16">
              THE DAMAGE
            </GlitchHeading>
          </Reveal>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-[1px] bg-white/10">
            {[
              { value: 25, suffix: "%", label: "OF SOLANA'S COMPUTE", sublabel: "consumed per sentence" },
              { value: 1.77, suffix: "B", label: "COMPUTE UNITS", sublabel: "per sentence generated" },
              { value: 2083, suffix: "", label: "TRANSACTIONS", sublabel: "for 9 tokens of output" },
              { value: 148, suffix: "", label: "BLOCKS CONSUMED", sublabel: "~59 seconds of chain" },
            ].map((stat, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="bg-black p-8 md:p-12 group hover:bg-white/[0.03] transition-colors duration-500">
                  <p className="text-4xl md:text-6xl lg:text-7xl font-bold text-[#00F0FF] mb-4">
                    <Counter end={stat.value} suffix={stat.suffix} />
                  </p>
                  <p className="font-mono text-xs tracking-widest text-white/70 mb-1">{stat.label}</p>
                  <p className="font-mono text-xs text-white/50">{stat.sublabel}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ HOW IT WORKS ═════════════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-start">
            <div>
              <Reveal>
                <p className="font-mono text-[#00F0FF] text-xs tracking-widest mb-6">
                  // ARCHITECTURE
                </p>
                <GlitchHeading className="text-4xl md:text-6xl lg:text-7xl mb-8">
                  HOW WE DID IT
                </GlitchHeading>
              </Reveal>

              <Reveal delay={0.2}>
                <p className="text-white/80 text-lg leading-relaxed mb-8">
                  A full GPT-Neo transformer with 8 attention layers, 16 heads, and a 50,257-token
                  vocabulary — deployed as a Solana BPF program. Every forward pass happens on-chain.
                  No oracles. No off-chain compute. Pure blockchain inference.
                </p>
              </Reveal>

              <Reveal delay={0.3}>
                <div className="space-y-4 font-mono text-sm">
                  {[
                    { label: "MODEL", value: "TinyStories-1M (GPT-Neo)" },
                    { label: "WEIGHTS", value: "10.78 MB across 2 accounts" },
                    { label: "PRECISION", value: "F16 embed + F32 layers + INT8 lm_head" },
                    { label: "LAYERS", value: "8 transformer blocks, 16 attention heads" },
                    { label: "VOCAB", value: "50,257 tokens (full GPT-2)" },
                    { label: "INSTRUCTIONS", value: "15 per position per layer" },
                    { label: "CU LIMIT", value: "1.4M per transaction" },
                  ].map((item, i) => (
                    <div key={i} className="flex border-b border-white/5 pb-3">
                      <span className="text-white/50 w-40 shrink-0">{item.label}</span>
                      <span className="text-[#00F0FF]">{item.value}</span>
                    </div>
                  ))}
                </div>
              </Reveal>
            </div>

            <Reveal delay={0.2}>
              <div className="relative">
                <img
                  src={NEURAL_IMG}
                  alt="Neural network on blockchain"
                  className="w-full border border-white/10"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <p className="absolute bottom-4 left-4 font-mono text-xs text-white/50">
                  FIG.01 — NEURAL NETWORK ON CHAIN
                </p>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═══ BLOCK SPACE ANALYSIS ═════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16">
          <Reveal>
            <p className="font-mono text-[#FF2020] text-xs tracking-widest mb-6">
              // NETWORK IMPACT ANALYSIS
            </p>
            <GlitchHeading className="text-4xl md:text-7xl lg:text-8xl mb-12">
              BLOCK SPACE
            </GlitchHeading>
          </Reveal>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
            <Reveal delay={0.1}>
              <div className="relative">
                <img
                  src={BLOCK_IMG}
                  alt="Block space visualization"
                  className="w-full border border-white/10"
                />
                <div className="absolute inset-0 bg-gradient-to-r from-black/40 to-transparent" />
                <p className="absolute bottom-4 left-4 font-mono text-xs text-white/50">
                  FIG.02 — BLOCK SPACE CONSUMPTION
                </p>
              </div>
            </Reveal>

            <div>
              <Reveal delay={0.2}>
                <p className="text-white/80 text-lg leading-relaxed mb-8">
                  Solana processes 48 million compute units per block, with blocks every 400ms.
                  Each token of our LLM output requires ~148.5 million CU — that's 3 entire blocks
                  worth of compute. But the real constraint is the per-account write lock: only
                  12M CU per account per block.
                </p>
              </Reveal>

              <Reveal delay={0.3}>
                <div className="border border-[#FF2020]/30 bg-[#FF2020]/5 p-6 mb-8">
                  <p className="font-mono text-[#FF2020] text-sm font-bold mb-3">
                    CRITICAL FINDING
                  </p>
                  <p className="text-white/80 text-sm leading-relaxed">
                    One user generating a 9-token sentence would consume approximately 25% of
                    Solana's entire network compute capacity for one full minute. Multiple concurrent
                    users running inference would create significant block space contention.
                  </p>
                </div>
              </Reveal>

              <Reveal delay={0.4}>
                <div className="space-y-3 font-mono text-sm">
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Block CU limit</span>
                    <span className="text-white">48,000,000</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Write-lock limit/block</span>
                    <span className="text-white">12,000,000</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">CU per token</span>
                    <span className="text-[#FF2020]">~148,500,000</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Blocks per token</span>
                    <span className="text-[#FF2020]">~15 (write-lock limited)</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Wall time per token</span>
                    <span className="text-[#00F0FF]">~6 seconds (theoretical)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Cost per token</span>
                    <span className="text-[#00F0FF]">~0.001 SOL (~$0.17)</span>
                  </div>
                </div>
              </Reveal>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ EDITION 1 — CLOCKWORK ════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32 bg-white/[0.01]">
        <div className="px-4 md:px-8 lg:px-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-20 items-start">
            <div>
              <Reveal>
                <div className="inline-block bg-white text-black font-bold text-xs px-3 py-1 mb-6 tracking-wider">
                  PREVIOUSLY
                </div>
                <GlitchHeading className="text-4xl md:text-6xl lg:text-7xl mb-8">
                  EDITION 1
                </GlitchHeading>
              </Reveal>

              <Reveal delay={0.2}>
                <p className="text-white/80 text-lg leading-relaxed mb-6">
                  In 2023, I discovered that Clockwork's scheduling software allowed recursive
                  transactions — a transaction that spawns another transaction in the same slot.
                  With enough SOL to pay the fees, this created an infinite loop that overwhelmed
                  validators.
                </p>
              </Reveal>

              <Reveal delay={0.3}>
                <p className="text-white/80 text-lg leading-relaxed mb-6">
                  The Clockwork team shrugged it off. A few days later, the entire Solana network
                  went down when Clockwork came back online. They eventually shut down in October 2023,
                  citing "limited commercial upside."
                </p>
              </Reveal>

              <Reveal delay={0.4}>
                <blockquote className="border-l-2 border-[#00F0FF] pl-6 my-8">
                  <p className="text-white/80 text-lg italic leading-relaxed">
                    "I figured out that you could do recursive transactions. A transaction that calls
                    another transaction in the same slot. If you have enough money to pay the Pied Piper,
                    that's terrible for blockchains."
                  </p>
                  <footer className="mt-4 font-mono text-sm text-[#00F0FF]">
                    — staccoverflow, Darknet Diaries EP 152
                  </footer>
                </blockquote>
              </Reveal>

              <Reveal delay={0.5}>
                <div className="flex flex-wrap gap-3 mt-8">
                  <a
                    href="https://darknetdiaries.com/episode/152/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 border border-white/20 px-4 py-2 font-mono text-sm text-white/60 hover:text-[#00F0FF] hover:border-[#00F0FF] transition-colors"
                  >
                    LISTEN: DARKNET DIARIES EP 152
                    <span className="text-[#00F0FF]">&rarr;</span>
                  </a>
                </div>
              </Reveal>
            </div>

            <Reveal delay={0.2}>
              <div className="relative">
                <img
                  src={CLOCKWORK_IMG}
                  alt="Clockwork recursive exploit visualization"
                  className="w-full border border-white/10"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                <p className="absolute bottom-4 left-4 font-mono text-xs text-white/50">
                  FIG.03 — RECURSIVE TRANSACTION EXPLOIT
                </p>

                {/* Timeline */}
                <div className="mt-6 space-y-3 font-mono text-xs">
                  <div className="flex gap-4 items-start">
                    <span className="text-[#00F0FF] shrink-0 w-16">2023</span>
                    <span className="text-white/60">Clockwork recursive transaction exploit discovered</span>
                  </div>
                  <div className="flex gap-4 items-start">
                    <span className="text-[#FF2020] shrink-0 w-16">2023</span>
                    <span className="text-white/60">Solana network outage when Clockwork restarted</span>
                  </div>
                  <div className="flex gap-4 items-start">
                    <span className="text-white/50 shrink-0 w-16">OCT 23</span>
                    <span className="text-white/60">Clockwork shuts down permanently</span>
                  </div>
                  <div className="flex gap-4 items-start">
                    <span className="text-[#00F0FF] shrink-0 w-16">2026</span>
                    <span className="text-[#00F0FF]">Edition 2: On-chain LLM inference</span>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═══ THE CHALLENGE ════════════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16">
          <Reveal>
            <p className="font-mono text-[#00F0FF] text-xs tracking-widest mb-6">
              // OPEN INVITATION
            </p>
          </Reveal>

          <Reveal delay={0.1}>
            <h2 className="text-5xl md:text-8xl lg:text-[10rem] font-bold tracking-tighter leading-[0.85] mb-12">
              YOUR<br />
              <span className="text-[#00F0FF]">TURN</span>
            </h2>
          </Reveal>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
            <Reveal delay={0.2}>
              <p className="text-white/80 text-xl leading-relaxed mb-8">
                The program is deployed. The weights are on-chain. The game is live.
                Install the CLI, pick a prompt, and try to make the LLM say "magic."
                The pot grows with every play. Can you crack the neural network?
              </p>

              <p className="text-white/80 text-xl leading-relaxed mb-8">
                This isn't just a game — it's a stress test. Every play sends ~200 transactions
                through Solana's scheduler. Multiple concurrent players create real block space
                contention. Let's see what breaks first.
              </p>

              {/* NPM Package Card */}
              <div className="border border-[#00F0FF]/30 bg-[#00F0FF]/5 p-6">
                <p className="font-mono text-[#00F0FF] text-sm mb-3">NPM PACKAGE</p>
                <div className="bg-black border border-white/10 p-4 font-mono text-sm mb-4">
                  <p className="text-white/50">$ npm install -g solana-llm</p>
                </div>
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Version</span>
                    <span className="text-[#00F0FF]">1.1.0</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Binaries</span>
                    <span className="text-white">solana-llm, solana-llm-game</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 pb-2">
                    <span className="text-white/60">Size</span>
                    <span className="text-white">340 KB (includes vocab)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Dependency</span>
                    <span className="text-white">@solana/web3.js</span>
                  </div>
                </div>
                <a
                  href="https://www.npmjs.com/package/solana-llm"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 mt-4 font-mono text-sm text-[#00F0FF] hover:text-white transition-colors"
                >
                  npmjs.com/package/solana-llm →
                </a>
              </div>
            </Reveal>

            <Reveal delay={0.3}>
              <div className="space-y-6">
                <div className="border border-[#FF2020]/30 p-6 bg-[#FF2020]/5 hover:border-[#FF2020]/50 transition-colors">
                  <p className="font-mono text-[#FF2020] text-sm mb-3">PLAY THE GAME</p>
                  <p className="text-white font-bold text-lg mb-2">Say The Magic Word</p>
                  <p className="text-white/60 text-sm mb-3">
                    1 SOL entry. Pick a prompt (no magic words allowed!). If the on-chain LLM outputs " magic" (token 5536),
                    you win the entire pot. 20% dev fee apes & burns the token.
                  </p>
                  <div className="bg-black border border-white/10 p-3 font-mono text-xs">
                    <p className="text-[#00F0FF]">$ solana-llm-game -c config.json "Tell me a story"</p>
                  </div>
                </div>

                <div className="border border-white/10 p-6 hover:border-[#00F0FF]/30 transition-colors">
                  <p className="font-mono text-[#00F0FF] text-sm mb-3">RUN INFERENCE</p>
                  <p className="text-white font-bold text-lg mb-2">Direct LLM Access</p>
                  <p className="text-white/60 text-sm mb-3">
                    Run the full GPT-2 model on-chain without the game wrapper.
                    Generate up to 20 tokens from any prompt.
                  </p>
                  <div className="bg-black border border-white/10 p-3 font-mono text-xs">
                    <p className="text-[#00F0FF]">$ solana-llm -c config.json "Once upon a time"</p>
                  </div>
                </div>

                <div className="border border-white/10 p-6 hover:border-[#00F0FF]/30 transition-colors">
                  <p className="font-mono text-white/60 text-sm mb-3">USE AS LIBRARY</p>
                  <p className="text-white font-bold text-lg mb-2">Build On Top</p>
                  <p className="text-white/60 text-sm mb-3">
                    Import constants, token decoder, and helpers into your own project.
                    Full GPT-2 vocab included.
                  </p>
                  <div className="bg-black border border-white/10 p-3 font-mono text-xs">
                    <p className="text-[#00F0FF]">const {'{'} decodeToken, MAGIC_TOKEN {'}'} = require('solana-llm');</p>
                  </div>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═══ TECHNICAL DEEP DIVE ═════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32 bg-white/[0.01]">
        <div className="px-4 md:px-8 lg:px-16 max-w-4xl">
          <Reveal>
            <p className="font-mono text-white/50 text-xs tracking-widest mb-6">
              // FOR THE TECHNICALLY CURIOUS
            </p>
            <GlitchHeading className="text-4xl md:text-6xl mb-12">
              UNDER THE HOOD
            </GlitchHeading>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="space-y-8 text-white/80 text-base leading-relaxed">
              <p>
                The biggest challenge wasn't deploying the model — it was fitting each operation
                within Solana's 1.4 million compute unit limit per transaction. A single 64x64
                matrix multiplication in f32 costs ~700,000 CU in BPF bytecode. Each transformer
                layer requires Q, K, V, and O projections plus a 4x-wide FFN — that's 8 matrix
                multiplications per layer, per position.
              </p>

              <p>
                We split each layer into <span className="text-[#00F0FF] font-mono">15 separate instructions</span>:
                LN1, Q_PROJ, K_PROJ, ATTN, V_O_PROJ, LN2, and 4 FFN chunks (UP_A, UP_B, DOWN_A, DOWN_B)
                plus GELU activation. For a 4-token prompt through 8 layers, that's 480 transactions
                just for the prefill phase.
              </p>

              <div className="border border-white/10 bg-black p-6 font-mono text-sm overflow-x-auto">
                <p className="text-white/50 mb-3">// Per-token instruction breakdown</p>
                <p><span className="text-[#00F0FF]">EMBED</span>          <span className="text-white/50">→</span> 1 tx   <span className="text-white/50">// token embedding lookup</span></p>
                <p><span className="text-[#00F0FF]">LAYER x8</span>       <span className="text-white/50">→</span> 128 tx <span className="text-white/50">// 16 instructions per layer</span></p>
                <p><span className="text-[#00F0FF]">OUTPUT_LN</span>      <span className="text-white/50">→</span> 1 tx   <span className="text-white/50">// final layer norm</span></p>
                <p><span className="text-[#00F0FF]">COPY_HIDDEN</span>    <span className="text-white/50">→</span> 1 tx   <span className="text-white/50">// prepare for argmax</span></p>
                <p><span className="text-[#00F0FF]">ARGMAX</span>         <span className="text-white/50">→</span> 64 tx  <span className="text-white/50">// 4 workers x 16 sub-chunks</span></p>
                <p><span className="text-[#00F0FF]">MERGE</span>          <span className="text-white/50">→</span> 1 tx   <span className="text-white/50">// find best token</span></p>
                <p className="text-white/50 mt-3 border-t border-white/10 pt-3">TOTAL: ~191 transactions per generated token</p>
              </div>

              <p>
                The quantization story is equally wild. INT8 quantization was too lossy for a 64-dimension
                model — cosine similarity between INT8 and f32 hidden states degraded to 0.287 by layer 8.
                We ended up with a hybrid approach: f16 embeddings, f32 transformer weights, and INT8 only
                for the output projection (lm_head). The model produces output that matches the HuggingFace
                reference token-for-token.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ MULTI-USER ══════════════════════════════════ */}
      <section className="relative border-t border-white/10 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16">
          <Reveal>
            <GlitchHeading className="text-4xl md:text-7xl lg:text-8xl mb-12">
              MULTI-USER
            </GlitchHeading>
          </Reveal>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-[1px] bg-white/10">
            {[
              {
                title: "PER-USER STATE",
                value: "141 KB",
                detail: "Each wallet gets its own state account with KV cache for context. Fully independent — your inference doesn't touch anyone else's.",
              },
              {
                title: "RENT DEPOSIT",
                value: "~1.01 SOL",
                detail: "Goes to the prize pot when you close your session. The state account stores hidden states, KV cache across all 8 layers, and worker accounts.",
              },
              {
                title: "CONTEXT LENGTH",
                value: "32 TOKENS",
                detail: "Per-layer KV cache supports up to 32 positions. The model remembers your full conversation context for multi-turn generation.",
              },
            ].map((item, i) => (
              <Reveal key={i} delay={i * 0.1}>
                <div className="bg-black p-8 md:p-12">
                  <p className="font-mono text-xs text-white/60 tracking-widest mb-4">{item.title}</p>
                  <p className="text-3xl md:text-5xl font-bold text-[#00F0FF] mb-4">{item.value}</p>
                  <p className="text-white/60 text-sm leading-relaxed">{item.detail}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ CTA / FOOTER ════════════════════════════════ */}
      <section className="relative border-t border-[#00F0FF]/20 py-20 md:py-32">
        <div className="px-4 md:px-8 lg:px-16 text-center">
          <Reveal>
            <p className="font-mono text-[#00F0FF] text-sm tracking-widest mb-8">
              &gt; READY TO STRESS TEST?_
            </p>
            <h2 className="text-4xl md:text-7xl font-bold tracking-tighter mb-8">
              THE CODE IS <span className="text-[#00F0FF]">OPEN</span>
            </h2>
            <p className="text-white/70 text-lg max-w-2xl mx-auto mb-12">
              Everything — the Solana program, the game client, the npm package —
              is available for anyone to install, play, and stress-test on mainnet.
              Say the magic word. Win the pot. Dev fee apes & burns the token.
            </p>
          </Reveal>

          <Reveal delay={0.2}>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="https://github.com/staccoverflow"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-3 bg-[#00F0FF] text-black font-bold px-8 py-4 text-lg hover:bg-white transition-colors animate-pulse-glow"
              >
                VIEW SOURCE CODE
              </a>
              <a
                href="https://x.com/STACCoverflow"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-3 border border-white/20 text-white font-bold px-8 py-4 text-lg hover:border-[#00F0FF] hover:text-[#00F0FF] transition-colors"
              >
                FOLLOW @STACCoverflow
              </a>
            </div>
          </Reveal>

          {/* CTA buttons */}
          <Reveal delay={0.3}>
            <div className="mt-16 max-w-3xl mx-auto">
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <a
                  href="https://www.npmjs.com/package/solana-llm"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 bg-[#FF2020] text-white font-bold px-8 py-4 text-lg hover:bg-[#FF4040] transition-colors"
                >
                  INSTALL: npm i -g solana-llm →
                </a>
                <a
                  href="https://axiom.trade/@raycc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 border border-white/20 text-white font-bold px-8 py-4 text-lg hover:border-[#00F0FF] hover:text-[#00F0FF] transition-colors"
                >
                  BUY TOKEN ON AXIOM
                </a>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═══ DISCLAIMER ═════════════════════════════════ */}
      <section className="border-t border-white/5 py-10 px-4 md:px-8 lg:px-16">
        <div className="max-w-3xl mx-auto">
          <p className="font-mono text-white/40 text-[10px] tracking-widest mb-3">// DISCLAIMER</p>
          <p className="font-mono text-white/50 text-xs leading-relaxed">
            Honest disclaimer: We deployed a full neural network on a blockchain and turned it into a game.
            That's not what blockchains are for. The game costs 1 SOL to play — 0.8 SOL goes to the prize pot,
            0.2 SOL dev fee apes & burns the token on Axiom. 9 magic-related tokens are banned from prompts on-chain.
            This is not financial advice. No promises, no guarantees, no refunds. Just vibes, math, and an
            unreasonable number of transactions.
          </p>
        </div>
      </section>

      {/* ═══ FOOTER ══════════════════════════════════════ */}
      <footer className="border-t border-white/5 py-8 px-4 md:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4 font-mono text-xs text-white/40">
          <span>BREAK SOLANA: EDITION 2 // STACCOVERFLOW // 2026</span>
          <span>
            EDITION 1: <a href="https://darknetdiaries.com/episode/152/" target="_blank" rel="noopener noreferrer" className="text-white/50 hover:text-[#00F0FF] transition-colors">CLOCKWORK.XYZ EXPLOIT (2023)</a>
          </span>
          <span>BUILT WITH SPITE AND COMPUTE UNITS</span>
        </div>
      </footer>
    </div>
  );
}
