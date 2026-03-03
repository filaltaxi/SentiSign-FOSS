import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'] as const;
export type EmotionType = typeof EMOTIONS[number];

interface EmotionStripProps {
    currentEmotion: EmotionType;
    onSelectEmotion?: (emotion: EmotionType) => void;
}

export function EmotionStrip({ currentEmotion, onSelectEmotion }: EmotionStripProps) {
    const isInteractive = typeof onSelectEmotion === 'function';

    return (
        <div className="flex flex-wrap gap-1.5">
            {EMOTIONS.map((emo) => (
                <button
                    key={emo}
                    type="button"
                    onClick={() => onSelectEmotion?.(emo)}
                    className={twMerge(
                        clsx(
                            "rounded-full px-3.5 py-1.5 text-[0.72rem] font-bold uppercase tracking-[0.14em] shadow-sm transition-all duration-300",
                            isInteractive && "cursor-pointer hover:-translate-y-0.5 hover:shadow-[0_10px_16px_rgba(15,34,68,0.14)]",
                            emo === currentEmotion
                                ? "scale-[1.06] border border-[#99c8ff] bg-gradient-to-r from-[rgba(51,153,255,0.18)] to-[rgba(0,127,255,0.18)] text-brand shadow-[0_8px_20px_rgba(0,127,255,0.25)]"
                                : "border border-border-color bg-white text-muted"
                        )
                    )}
                >
                    {emo}
                </button>
            ))}
        </div>
    );
}
