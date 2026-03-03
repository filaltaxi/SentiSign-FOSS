import { motion, AnimatePresence } from 'framer-motion';

interface WordBufferProps {
    words: string[];
}

export function WordBuffer({ words }: WordBufferProps) {
    return (
        <div className="min-h-[56px] rounded-2xl border border-border-color bg-[#f8fbff] p-3 flex flex-wrap items-center gap-2 [@media(max-height:820px)]:min-h-[44px] [@media(max-height:820px)]:p-2.5">
            {words.length === 0 ? (
                <span className="text-muted text-[0.82rem] italic font-medium tracking-wide">
                    Words will appear here as you sign...
                </span>
            ) : (
                <AnimatePresence>
                    {words.map((word, i) => (
                        <motion.span
                            key={`${word}-${i}`}
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            transition={{ duration: 0.2 }}
                            className="block rounded-full border border-[#c8ddff] bg-white px-3 py-1.5 text-[0.74rem] font-bold uppercase tracking-[0.12em] text-brand shadow-[0_6px_16px_rgba(0,127,255,0.15)]"
                        >
                            {word}
                        </motion.span>
                    ))}
                </AnimatePresence>
            )}
        </div>
    );
}
