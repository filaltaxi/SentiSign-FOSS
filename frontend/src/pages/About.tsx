import { motion } from 'framer-motion';

export function About() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="px-5 md:px-10 py-12 max-w-[860px] mx-auto min-h-[85vh]"
        >
            <h1 className="font-heading font-extrabold text-[clamp(2.2rem,5vw,3.5rem)] leading-[1.1] pb-[0.08em] mb-4 tracking-tight">
                About <em className="text-brand not-italic">SentiSign</em>
            </h1>
            <p className="text-muted text-[1.1rem] leading-relaxed mb-12 max-w-[650px]">
                SentiSign is a multimodal assistive communication system that translates ASL hand signs and facial emotion into natural, emotionally-expressive speech &mdash; in real time, on a standard webcam.
            </p>

            <div className="mb-14">
                <h2 className="font-heading font-bold text-[1.2rem] text-brand uppercase tracking-widest mb-6">How It Works</h2>
                <div className="flex flex-col gap-0 border-l border-border-color ml-2 pl-6">

                    <div className="relative py-6 border-b border-border-color/50 last:border-0 group">
                        <div className="absolute -left-[37px] top-6 w-8 h-8 rounded-full bg-surface border border-border-color flex items-center justify-center font-heading font-bold text-[1rem] text-muted group-hover:text-brand transition-colors shadow-sm">
                            1
                        </div>
                        <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-[1.1rem] text-text">Sign Recognition</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(51,153,255,0.12)] border border-brand/20 text-brand font-medium">MediaPipe + MLP</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            MediaPipe Hands extracts 21 three-dimensional landmarks per hand (126 features for two hands). A lightweight MLP classifier maps normalised landmark coordinates to vocabulary words. Background, lighting, and skin tone are irrelevant &mdash; only hand geometry matters.
                        </p>
                    </div>

                    <div className="relative py-6 border-b border-border-color/50 last:border-0 group">
                        <div className="absolute -left-[37px] top-6 w-8 h-8 rounded-full bg-surface border border-border-color flex items-center justify-center font-heading font-bold text-[1rem] text-muted group-hover:text-amber transition-colors shadow-sm">
                            2
                        </div>
                        <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-[1.1rem] text-text">Emotion Recognition</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(255,179,71,0.1)] border border-amber/20 text-amber font-medium">ResNet CNN</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            Simultaneously, a ResNet-based CNN detects the signer's facial emotion across seven universal categories: angry, disgust, fear, happy, neutral, sad, surprise. Both recognition tasks run in the same webcam session with no interruption.
                        </p>
                    </div>

                    <div className="relative py-6 border-b border-border-color/50 last:border-0 group">
                        <div className="absolute -left-[37px] top-6 w-8 h-8 rounded-full bg-surface border border-border-color flex items-center justify-center font-heading font-bold text-[1rem] text-muted group-hover:text-brand transition-colors shadow-sm">
                            3
                        </div>
                        <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-[1.1rem] text-text">Sentence Generation</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(51,153,255,0.12)] border border-brand/20 text-brand font-medium">Flan-T5-Large</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            Recognised words are passed to Flan-T5-Large, an instruction-tuned language model that converts telegraphic word sequences into grammatically complete sentences &mdash; zero-shot, no fine-tuning required.
                        </p>
                    </div>

                    <div className="relative py-6 border-b border-border-color/50 last:border-0 group">
                        <div className="absolute -left-[37px] top-6 w-8 h-8 rounded-full bg-surface border border-border-color flex items-center justify-center font-heading font-bold text-[1rem] text-muted group-hover:text-amber transition-colors shadow-sm">
                            4
                        </div>
                        <div className="flex items-center gap-3 mb-2">
                            <h3 className="font-semibold text-[1.1rem] text-text">Emotion-Aware Speech</h3>
                            <span className="text-[0.7rem] px-2 py-0.5 rounded-md bg-[rgba(255,179,71,0.1)] border border-amber/20 text-amber font-medium">Cartesia default</span>
                        </div>
                        <p className="text-muted text-[0.95rem] leading-relaxed max-w-[600px]">
                            The sentence is synthesised with Cartesia Sonic-3 by default, with an optional Chatterbox fallback selectable through the backend environment. That keeps Cartesia as the main path while still allowing a local fallback engine.
                        </p>
                    </div>

                </div>
            </div>

            <div className="mb-14">
                <h2 className="font-heading font-bold text-[1.2rem] text-brand uppercase tracking-widest mb-6">Team</h2>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-3">
                    {[
                        { name: 'Omar Sheriff', role: 'Student, ECE' },
                        { name: 'P S Arjun', role: 'Student, ECE' },
                        { name: 'Sree Harinandan', role: 'Student, ECE' },
                        { name: 'Sudipto Bagchi', role: 'Student, ECE' },
                    ].map(member => (
                        <div key={member.name} className="bg-surface border border-border-color p-4 rounded-xl flex flex-col justify-center">
                            <span className="font-heading font-bold text-[0.95rem] text-text">{member.name}</span>
                            <span className="text-[0.8rem] text-muted">{member.role}</span>
                        </div>
                    ))}
                </div>
                <div className="bg-surface border border-[rgba(51,153,255,0.35)] p-4 rounded-xl w-full sm:w-[calc(50%-8px)] mb-3">
                    <span className="font-heading font-bold text-[0.95rem] text-brand">Jasmin Sebastin</span>
                    <span className="block text-[0.8rem] text-muted">Guide, Asst. Professor</span>
                </div>
                <p className="text-muted text-[0.85rem] mt-4 opacity-80">Rajagiri School of Engineering and Technology, Kerala, India</p>
            </div>

            <div>
                <h2 className="font-heading font-bold text-[1.2rem] text-brand uppercase tracking-widest mb-6">Technology Stack</h2>
                <div className="flex flex-wrap gap-2.5">
                    {['Python 3.10', 'FastAPI', 'PyTorch 2.6 + CUDA', 'MediaPipe 0.10.9', 'MediaPipe.js', 'Flan-T5-Large', 'Cartesia + Chatterbox', 'ResNet CNN', 'scikit-learn', 'OpenCV'].map(tech => (
                        <div key={tech} className="bg-surface border border-border-color rounded-lg px-3.5 py-1.5 text-[0.85rem] text-muted shadow-sm">
                            {tech}
                        </div>
                    ))}
                </div>
            </div>

        </motion.div>
    );
}
