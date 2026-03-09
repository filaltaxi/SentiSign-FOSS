import { motion, AnimatePresence } from 'framer-motion';

interface WordBufferProps {
    words: string[];
}

export function WordBuffer({ words }: WordBufferProps) {
    return (
        <div className="h-full min-h-[56px] rounded-[24px] border border-[#bfd8ff] bg-[linear-gradient(180deg,#f8fbff_0%,#eef5ff_100%)] px-4 py-3 shadow-[0_16px_34px_rgba(15,34,68,0.14),inset_0_1px_0_rgba(255,255,255,0.86)] [@media(max-height:820px)]:px-3 [@media(max-height:820px)]:py-2.5">
            <div className="mb-2 flex items-center justify-between gap-3">
                <span className="font-mono text-[0.62rem] font-bold uppercase tracking-[0.2em] text-[#5a83b4]">
                    Live Words
                </span>
                <span className="rounded-full border border-[#cfe2ff] bg-white/88 px-2.5 py-1 font-mono text-[0.58rem] font-bold uppercase tracking-[0.16em] text-[#3b77b5] shadow-[0_6px_14px_rgba(0,127,255,0.08)]">
                    {words.length} token{words.length === 1 ? '' : 's'}
                </span>
            </div>

            <div className="flex min-h-0 flex-wrap content-start gap-2.5">
                {words.length === 0 ? (
                    <span className="pt-1 text-[0.84rem] font-medium italic tracking-wide text-[#6b8db7]">
                        Words will appear here as you sign...
                    </span>
                ) : (
                    <AnimatePresence>
                        {words.map((word, i) => (
                            <motion.span
                                key={`${word}-${i}`}
                                initial={{ opacity: 0, scale: 0.86, y: 6 }}
                                animate={{ opacity: 1, scale: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.86 }}
                                transition={{ duration: 0.2 }}
                                className="block rounded-full border border-[#c0d8ff] bg-white px-3.5 py-2 text-[0.76rem] font-extrabold uppercase tracking-[0.12em] text-brand shadow-[0_10px_18px_rgba(0,127,255,0.14)]"
                            >
                                {word}
                            </motion.span>
                        ))}
                    </AnimatePresence>
                )}
            </div>
        </div>
    );
}
