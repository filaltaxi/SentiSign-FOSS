import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export function SignsGallery() {
    const [signs, setSigns] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [query, setQuery] = useState('');

    useEffect(() => {
        fetch('/api/signs')
            .then(r => r.json())
            .then(data => {
                setSigns(data.signs || []);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, []);

    const filtered = useMemo(() => {
        const q = query.trim().toUpperCase();
        if (!q) return signs;
        return signs.filter(s => s.word.includes(q) || s.class.includes(q));
    }, [signs, query]);

    return (
        <div className="px-5 md:px-10 py-12 max-w-[1200px] mx-auto animate-in fade-in zoom-in-95 duration-500 ease-out">

            <div className="mb-8">
                <h1 className="font-heading font-extrabold text-[clamp(2.2rem,5vw,3.5rem)] leading-[1.1] pb-[0.08em] mb-3 tracking-tight">
                    Signs <em className="text-brand not-italic">Gallery</em>
                </h1>
                <p className="text-muted text-[1.05rem] max-w-[500px] leading-relaxed">
                    All signs currently in the system. Hover a card to see the reference gesture. Use these to learn the vocabulary before signing.
                </p>
            </div>

            <div className="mb-8">
                <input
                    type="text"
                    placeholder="Search by word (e.g. HELP, WATER)..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-full max-w-[400px] bg-surface border border-border-color rounded-xl px-4 py-3 text-text font-sans outline-none focus:border-brand transition-colors placeholder:text-muted"
                />
            </div>

            <div className="text-[0.85rem] text-muted mb-4 pl-1">
                {loading ? 'Crunching numbers...' : `${filtered.length} sign${filtered.length !== 1 ? 's' : ''} available`}
            </div>

            {loading ? (
                <div className="py-20 text-center text-muted">
                    <span className="inline-block w-6 h-6 border-2 border-[rgba(0,0,0,0.3)] border-t-brand rounded-full animate-[spin_0.7s_linear_infinite]" />
                </div>
            ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                    <AnimatePresence>
                        {filtered.map(s => (
                            <motion.div
                                layout
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                transition={{ duration: 0.2 }}
                                key={s.class}
                                className="group bg-surface border border-border-color rounded-xl overflow-hidden relative transition-all duration-300 hover:border-brand hover:-translate-y-1 hover:shadow-[0_12px_24px_rgba(0,127,255,0.16)] cursor-pointer"
                            >
                                {s.class.startsWith('CUSTOM_') && (
                                    <div className="absolute top-2 right-2 z-10 bg-[#fff0e7] border border-[#ffc7a9] text-[#c85a21] rounded shadow-sm px-1.5 py-0.5 text-[0.6rem] font-bold tracking-wider">
                                        NEW
                                    </div>
                                )}
                                <div className="w-full aspect-square bg-[#edf5ff] relative flex items-center justify-center overflow-hidden border-b border-border-color">
                                    {s.gif_url ? (
                                        <img src={s.gif_url} alt={s.word} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" loading="lazy" />
                                    ) : (
                                        <div className="text-[3rem] text-border-color select-none">&#9995;</div>
                                    )}
                                </div>
                                <div className="p-3 bg-surface/80 backdrop-blur">
                                    <div className="font-heading font-bold text-[1rem] text-text truncate mb-0.5">{s.word}</div>
                                    <div className="text-[0.7rem] text-muted truncate">Sign: {s.class}</div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {filtered.length === 0 && (
                        <div className="col-span-full py-10 text-center text-muted">No signs found for that search.</div>
                    )}
                </div>
            )}

        </div>
    );
}
