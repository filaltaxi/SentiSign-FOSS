import { useEffect, useMemo, useRef, useState } from 'react';
import { Search, Sparkles, X, Image } from 'lucide-react';
import { useModel } from '../model/ModelContext';

type Sign = {
    word: string;
    class: string;
    gif_url?: string | null;
};

const normalize = (value: string) => value.trim().toLowerCase();

const scoreToken = (token: string, target: string): number | null => {
    if (!token) return 0;
    if (!target) return null;

    if (target === token) return 2_000;
    if (target.startsWith(token)) return 1_500 - Math.min(300, (target.length - token.length) * 4);

    const includesAt = target.indexOf(token);
    if (includesAt !== -1) return 1_150 - Math.min(500, includesAt * 10) - Math.min(250, (target.length - token.length) * 2);

    // Subsequence score (fuzzy): token chars must appear in order.
    let score = 520;
    let lastMatch = -2;
    let cursor = 0;

    for (const ch of token) {
        const next = target.indexOf(ch, cursor);
        if (next === -1) return null;

        const gap = next - cursor;
        const consecutive = next === lastMatch + 1;
        const wordStart = next === 0 || target[next - 1] === '_' || target[next - 1] === '-' || target[next - 1] === ' ';

        score += 40;
        score += consecutive ? 26 : 0;
        score += wordStart ? 22 : 0;
        score -= Math.min(18, gap * 2);

        lastMatch = next;
        cursor = next + 1;
    }

    score -= Math.min(260, target.length * 2);
    return score;
};

const rankSign = (sign: Sign, tokens: string[]) => {
    const word = normalize(sign.word);
    const cls = normalize(sign.class);

    let total = 0;
    let bestField: 'word' | 'class' = 'word';

    for (const token of tokens) {
        const wordScore = scoreToken(token, word);
        const classScore = scoreToken(token, cls);

        const tokenBest =
            wordScore === null && classScore === null
                ? null
                : Math.max(wordScore ?? -Infinity, classScore ?? -Infinity);

        if (tokenBest === null || tokenBest === -Infinity) return null;

        total += tokenBest;
        if ((classScore ?? -Infinity) > (wordScore ?? -Infinity)) bestField = 'class';
    }

    return { total, bestField };
};

function SignsSkeletonList() {
    return (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            {Array.from({ length: 16 }).map((_, idx) => (
                <div
                    key={idx}
                    className="relative aspect-square overflow-hidden rounded-2xl border border-border-color bg-white/70 shadow-[0_10px_24px_rgba(15,34,68,0.06)]"
                >
                    <div className="absolute inset-0 bg-[#edf5ff]" />
                    <div className="absolute inset-0 animate-pulse bg-[linear-gradient(110deg,rgba(255,255,255,0.0),rgba(255,255,255,0.75),rgba(255,255,255,0.0))] [background-size:320px_100%]" />
                    <div className="absolute inset-x-3 bottom-3">
                        <div className="h-4 w-[62%] animate-pulse rounded bg-white/70" />
                        <div className="mt-2 h-3 w-[82%] animate-pulse rounded bg-white/55" />
                    </div>
                </div>
            ))}
        </div>
    );
}

export function SignsGallery() {
    const { model } = useModel();
    const activeModel = model ?? 'mlp';
    const [signs, setSigns] = useState<Sign[]>([]);
    const [loading, setLoading] = useState(true);
    const [query, setQuery] = useState('');
    const [onlyNew, setOnlyNew] = useState(false);
    const [onlyWithGif, setOnlyWithGif] = useState(false);
    const searchRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        const controller = new AbortController();

        (async () => {
            try {
                const endpoint = activeModel === 'lstm' ? '/api/temporal/signs' : '/api/signs';
                const res = await fetch(endpoint, { signal: controller.signal });
                const data = await res.json() as { signs?: Sign[] };
                setSigns(Array.isArray(data.signs) ? data.signs : []);
            } catch (err) {
                if (err instanceof DOMException && err.name === 'AbortError') return;
                if (err instanceof Error && err.name === 'AbortError') return;
                console.error(err);
            } finally {
                setLoading(false);
            }
        })();

        return () => controller.abort();
    }, [activeModel]);

    useEffect(() => {
        if (activeModel === 'lstm') {
            setOnlyWithGif(false);
        }
    }, [activeModel]);

    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === '/' && (e.target as HTMLElement | null)?.tagName !== 'INPUT') {
                e.preventDefault();
                searchRef.current?.focus();
            }

            if (e.key === 'Escape') {
                setQuery('');
                searchRef.current?.blur();
            }
        };
        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, []);

    const tokens = useMemo(() => normalize(query).split(/\s+/).filter(Boolean), [query]);

    const filtered = useMemo(() => {
        const base = onlyNew
            ? signs.filter((s) => s.class.startsWith('CUSTOM_'))
            : signs;

        const scoped = activeModel === 'mlp' && onlyWithGif
            ? base.filter((s) => Boolean(s.gif_url))
            : base;

        if (tokens.length === 0) {
            const sorted = [...scoped].sort((a, b) => a.word.localeCompare(b.word));
            return sorted.map((sign) => ({ sign, score: 0, bestField: 'word' as const }));
        }

        const ranked = scoped
            .map((sign) => {
                const rank = rankSign(sign, tokens);
                if (!rank) return null;
                return { sign, score: rank.total, bestField: rank.bestField };
            })
            .filter((row): row is NonNullable<typeof row> => row !== null);

        ranked.sort((a, b) => {
            if (b.score !== a.score) return b.score - a.score;
            return a.sign.word.localeCompare(b.sign.word);
        });

        return ranked;
    }, [activeModel, onlyNew, onlyWithGif, signs, tokens]);

    const shownCountLabel = loading
        ? 'Loading signs…'
        : `${filtered.length} sign${filtered.length === 1 ? '' : 's'} shown`;

    return (
        <div className="mx-auto grid h-[calc(100dvh-var(--app-nav-h))] w-full max-w-[1400px] grid-cols-1 items-stretch gap-4 px-4 py-4 sm:px-8 lg:grid-cols-[minmax(440px,560px)_12px_1px_12px_minmax(0,1fr)] lg:gap-0 lg:overflow-hidden">
            <aside className="min-h-0 h-full">
                <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-[26px] border border-border-color bg-surface/95 shadow-[0_16px_36px_rgba(15,34,68,0.10)]">
                    <header className="relative overflow-hidden border-b border-border-color px-5 pb-4 pt-5">
                        <div className="pointer-events-none absolute -left-10 -top-12 h-[160px] w-[240px] rotate-[-12deg] rounded-[38px] bg-[radial-gradient(circle_at_30%_30%,rgba(0,127,255,0.22),transparent_60%),radial-gradient(circle_at_70%_70%,rgba(255,127,64,0.18),transparent_60%)] blur-[1px]" />
                        <div className="flex items-start justify-between gap-4">
                            <div>
                                <h1 className="font-heading text-[1.45rem] font-extrabold leading-[1.05] tracking-tight text-text">
                                    Find a <span className="text-brand">Sign</span>
                                </h1>
                                <p className="mt-2 max-w-[34ch] text-[0.9rem] leading-relaxed text-muted">
                                    Type to fuzzy-match <span className="font-bold text-text">Word</span> + <span className="font-bold text-text">Class</span>. Press <span className="font-bold text-text">/</span> to focus.
                                </p>
                            </div>
                            <div className="hidden rounded-2xl border border-[#d2e4ff] bg-[#edf5ff] px-3 py-2 text-[0.72rem] font-extrabold uppercase tracking-[0.16em] text-brand md:block">
                                Fuzzy
                            </div>
                        </div>
                    </header>

                    <div className="flex min-h-0 flex-1 flex-col gap-4 px-5 py-5">
                        <div className="relative">
                            <div className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-muted">
                                <Search size={18} />
                            </div>
                            <input
                                ref={searchRef}
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Search (e.g. help, water, custom_…)…"
                                className="h-12 w-full rounded-2xl border border-border-color bg-white pl-11 pr-11 text-[0.95rem] font-semibold text-text outline-none transition-all duration-200 placeholder:text-muted focus:border-brand focus:shadow-[0_12px_24px_rgba(0,127,255,0.14)]"
                                inputMode="search"
                            />
                            {query.trim().length > 0 && (
                                <button
                                    type="button"
                                    onClick={() => setQuery('')}
                                    className="absolute inset-y-0 right-3 my-auto flex h-8 w-8 items-center justify-center rounded-full border border-[#d2e4ff] bg-[#f3f8ff] text-brand transition-all duration-200 hover:bg-[#ecf4ff] hover:shadow-[0_10px_16px_rgba(0,127,255,0.12)]"
                                    aria-label="Clear search"
                                >
                                    <X size={16} />
                                </button>
                            )}
                        </div>

                        <div className="grid grid-cols-1 gap-2">
                            <button
                                type="button"
                                onClick={() => setOnlyNew((v) => !v)}
                                className={`flex items-center justify-between rounded-2xl border px-4 py-3 text-left text-[0.88rem] font-bold transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_12px_22px_rgba(15,34,68,0.08)] ${onlyNew
                                    ? 'border-[#ffc7a9] bg-[#fff1e8] text-[#c85a21]'
                                    : 'border-border-color bg-white text-text hover:border-[#ffd3bf]'
                                    }`}
                            >
                                <span className="flex items-center gap-2">
                                    <Sparkles size={16} />
                                    Only “NEW” (CUSTOM_)
                                </span>
                                <span className={`rounded-full px-2 py-0.5 text-[0.68rem] font-extrabold uppercase tracking-[0.14em] ${onlyNew ? 'bg-white/60' : 'bg-[#edf5ff] text-brand'}`}>
                                    {onlyNew ? 'ON' : 'OFF'}
                                </span>
                            </button>

                            {activeModel === 'mlp' && (
                                <button
                                    type="button"
                                    onClick={() => setOnlyWithGif((v) => !v)}
                                    className={`flex items-center justify-between rounded-2xl border px-4 py-3 text-left text-[0.88rem] font-bold transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_12px_22px_rgba(15,34,68,0.08)] ${onlyWithGif
                                        ? 'border-[#bfdbff] bg-[#edf5ff] text-brand'
                                        : 'border-border-color bg-white text-text hover:border-[#b8d4ff]'
                                        }`}
                                >
                                    <span className="flex items-center gap-2">
                                        <Image size={16} />
                                        Only signs with GIFs
                                    </span>
                                    <span className={`rounded-full px-2 py-0.5 text-[0.68rem] font-extrabold uppercase tracking-[0.14em] ${onlyWithGif ? 'bg-white/60' : 'bg-[#edf5ff] text-brand'}`}>
                                        {onlyWithGif ? 'ON' : 'OFF'}
                                    </span>
                                </button>
                            )}
                        </div>

                        <div className="rounded-2xl border border-border-color bg-[#f9fbff] p-4">
                            <div className="text-[0.86rem] font-semibold text-muted">
                                {shownCountLabel}
                            </div>
                        </div>

                        <div className="mt-auto rounded-2xl border border-border-color bg-white/70 p-4">
                            <div className="text-[0.72rem] font-extrabold uppercase tracking-[0.16em] text-muted">
                                Tips
                            </div>
                            <ul className="mt-2 space-y-1.5 text-[0.9rem] font-semibold text-text">
                                <li className="flex items-start gap-2">
                                    <span className="mt-[0.35em] h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                                    Search matches even if you miss characters.
                                </li>
                                <li className="flex items-start gap-2">
                                    <span className="mt-[0.35em] h-1.5 w-1.5 shrink-0 rounded-full bg-[#c85a21]" />
                                    Press <span className="rounded-md border border-border-color bg-white px-1.5 py-0.5 text-[0.78rem] font-extrabold">Esc</span> to clear.
                                </li>
                            </ul>
                        </div>
                    </div>
                </section>
            </aside>

            <div className="hidden lg:block" aria-hidden="true" />
            <div
                className="hidden lg:block h-full w-px bg-border-color shadow-[0_0_0_1px_rgba(255,255,255,0.55),0_0_18px_rgba(0,127,255,0.06)]"
                aria-hidden="true"
            />
            <div className="hidden lg:block" aria-hidden="true" />

            <main className="min-h-0 h-full">
                <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-[26px] border border-border-color bg-surface/95 shadow-[0_16px_36px_rgba(15,34,68,0.10)]">
                    <header className="border-b border-border-color px-5 pb-4 pt-5">
                        <div className="flex flex-wrap items-end justify-between gap-3">
                            <div>
                                <h2 className="font-heading text-[1.35rem] font-extrabold tracking-tight text-text">
                                    Signs <span className="text-brand-end">Library</span>
                                </h2>
                                <p className="mt-2 max-w-[70ch] text-[0.92rem] leading-relaxed text-muted">
                                    Browse the vocabulary currently in the system. Cards are stable (no jumpy layout) and optimized for fast scrolling.
                                </p>
                            </div>

                            <div className="rounded-full border border-[#d2e4ff] bg-white px-3 py-1 text-[0.72rem] font-extrabold uppercase tracking-[0.16em] text-muted">
                                {loading ? 'Fetching…' : `${signs.length} total`}
                            </div>
                        </div>
                    </header>

                    <div className="min-h-0 flex-1 overflow-auto px-5 py-5 [scrollbar-gutter:stable]">
                        {loading ? (
                            <SignsSkeletonList />
                        ) : filtered.length === 0 ? (
                            <div className="grid place-items-center rounded-2xl border border-border-color bg-white/70 p-10 text-center">
                                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl border border-[#d2e4ff] bg-[#edf5ff] text-brand shadow-[0_14px_22px_rgba(0,127,255,0.12)]">
                                    <Search size={20} />
                                </div>
                                <div className="mt-4 font-heading text-[1.2rem] font-extrabold text-text">No matches</div>
                                <div className="mt-2 max-w-[42ch] text-[0.95rem] font-semibold text-muted">
                                    Try fewer characters, or search by class name (e.g. <span className="font-extrabold text-text">CUSTOM_</span>).
                                </div>
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
                                {filtered.map(({ sign, bestField, score }) => {
                                    const isNew = sign.class.startsWith('CUSTOM_');
                                    const primary = bestField === 'class' ? sign.class : sign.word;
                                    const secondary = bestField === 'class' ? sign.word : sign.class;

                                    return (
                                        <article
                                            key={sign.class}
                                            className="group relative aspect-square overflow-hidden rounded-2xl border border-border-color bg-white/75 shadow-[0_10px_24px_rgba(15,34,68,0.06)] transition-all duration-200 hover:-translate-y-0.5 hover:border-[#b8d4ff] hover:shadow-[0_18px_34px_rgba(0,127,255,0.12)]"
                                            style={{ contentVisibility: 'auto', containIntrinsicSize: '280px 280px' }}
                                        >
                                            <div className="absolute inset-0">
                                                {activeModel === 'mlp' && sign.gif_url ? (
                                                    <img
                                                        src={sign.gif_url}
                                                        alt={sign.word}
                                                        loading="lazy"
                                                        decoding="async"
                                                        className="h-full w-full object-cover transition-transform duration-700 group-hover:scale-110"
                                                    />
                                                ) : (
                                                    <div className="relative grid h-full w-full place-items-center bg-[radial-gradient(circle_at_30%_30%,rgba(0,127,255,0.22),transparent_56%),radial-gradient(circle_at_70%_70%,rgba(255,127,64,0.18),transparent_60%),linear-gradient(160deg,#f8fbff_0%,#eaf2ff_100%)]">
                                                        <div className="text-[3rem] text-[#93b7e6] [text-shadow:0_10px_18px_rgba(0,127,255,0.18)]">
                                                            {activeModel === 'lstm' ? 'LSTM' : '✋'}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>

                                            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-[58%] bg-[linear-gradient(180deg,rgba(15,34,68,0)_0%,rgba(15,34,68,0.55)_56%,rgba(15,34,68,0.82)_100%)]" />

                                            <div className="absolute left-4 top-4 flex flex-wrap items-center gap-2">
                                                {isNew && (
                                                    <div className="rounded-full border border-[#ffc7a9] bg-[#fff1e8] px-2 py-0.5 text-[0.62rem] font-extrabold uppercase tracking-[0.14em] text-[#c85a21] shadow-[0_10px_16px_rgba(200,90,33,0.12)]">
                                                        NEW
                                                    </div>
                                                )}
                                                {tokens.length > 0 && (
                                                    <div className="rounded-full border border-white/20 bg-black/35 px-2 py-0.5 text-[0.62rem] font-extrabold uppercase tracking-[0.14em] text-white/90 backdrop-blur">
                                                        {Math.max(0, Math.round(score / 10))}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="absolute inset-x-4 bottom-4">
                                                <div className="font-heading text-[clamp(0.9rem,1.55vw,1.04rem)] font-extrabold leading-[1.08] tracking-[-0.02em] text-white break-words [text-shadow:0_10px_22px_rgba(0,0,0,0.35)]">
                                                    {primary}
                                                </div>
                                                <div className="mt-1 text-[0.74rem] font-semibold leading-snug text-white/80 [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:2] overflow-hidden break-words">
                                                    {secondary}
                                                </div>
                                                <div className="mt-2 flex flex-wrap items-center gap-2">
                                                    <span className="rounded-full border border-white/15 bg-white/10 px-2 py-0.5 text-[0.62rem] font-extrabold uppercase tracking-[0.14em] text-white/90 backdrop-blur">
                                                        {bestField === 'class' ? 'Class match' : 'Word match'}
                                                    </span>
                                                    {activeModel === 'mlp' && (
                                                        <span className="rounded-full border border-white/15 bg-white/10 px-2 py-0.5 text-[0.62rem] font-extrabold uppercase tracking-[0.14em] text-white/90 backdrop-blur">
                                                            {sign.gif_url ? 'GIF' : 'NO GIF'}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </article>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </section>
            </main>
        </div>
    );
}
