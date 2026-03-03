import { motion } from 'framer-motion';
import { NavLink } from 'react-router-dom';

interface Gate4SuccessProps {
    onReset: () => void;
    word: string;
}

export function Gate4Success({ onReset, word }: Gate4SuccessProps) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-surface border border-brand/30 rounded-xl p-8 shadow-[0_18px_34px_rgba(0,127,255,0.12)] text-center max-w-[500px] mx-auto"
        >
            <div className="w-16 h-16 bg-[rgba(51,153,255,0.12)] border-2 border-brand rounded-full flex items-center justify-center text-brand text-[2rem] mx-auto mb-5 shadow-[0_0_15px_rgba(0,127,255,0.2)]">
                &#10003;
            </div>
            <h2 className="font-heading font-extrabold text-[1.8rem] mb-2 tracking-tight text-text">Success!</h2>
            <p className="text-muted mb-8 text-[1.05rem]">
                The sign model for <strong>{word}</strong> has been successfully trained and added to the vocabulary.
            </p>

            <div className="flex flex-col gap-3 w-full">
                <NavLink to="/signs" className="btn btn-primary w-full text-center flex items-center justify-center py-3.5 text-[1rem]">
                    View Signs Gallery
                </NavLink>
                <button onClick={onReset} className="btn btn-secondary w-full py-3.5 text-[1rem] border-border-color">
                    Add Another Sign
                </button>
            </div>
        </motion.div>
    );
}
