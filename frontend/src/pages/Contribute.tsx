import { useState } from 'react';
import { WizardLayout } from '../components/WizardLayout';
import { Gate1WordCheck } from '../components/Gate1WordCheck';
import { Gate2Gesture } from '../components/Gate2Gesture';
import { Gate3Record } from '../components/Gate3Record';
import { Gate4Success } from '../components/Gate4Success';
import { AnimatePresence } from 'framer-motion';

const STEPS = [
    { num: 1, label: 'Word Check' },
    { num: 2, label: 'Gesture Check' },
    { num: 3, label: 'Record Sign' },
    { num: 4, label: 'Done' }
];

export function Contribute() {
    const [currentStep, setCurrentStep] = useState(1);
    const [targetWord, setTargetWord] = useState('');

    const handleWordChecked = (word: string) => {
        setTargetWord(word);
        setCurrentStep(2);
    };

    const handleAddMoreSamples = (word: string) => {
        setTargetWord(word);
        setCurrentStep(3); // skip gesture check if adding samples to existing word
    };

    const handleGate2Back = () => setCurrentStep(1);
    const handleGate2Continue = () => setCurrentStep(3);

    const handleGate3Back = () => {
        setTargetWord('');
        setCurrentStep(1);
    };

    const handleGate3Submit = async (samples: any[], gifFrames: string[]) => {
        try {
            const res = await fetch('/api/signs/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    word: targetWord,
                    samples: samples,
                    gif_frames: gifFrames
                })
            });
            const data = await res.json();
            if (res.ok && data.status === 'success') {
                setCurrentStep(4);
            } else {
                throw new Error(data.detail || 'Submission failed');
            }
        } catch (err: any) {
            alert(`Error submitting sign: ${err.message}`);
        }
    };

    const handleReset = () => {
        setTargetWord('');
        setCurrentStep(1);
    };

    return (
        <div className="px-5 md:px-10 py-12 max-w-[860px] mx-auto min-h-[85vh]">
            <div className="mb-10 pl-1 animate-in fade-in zoom-in-95 duration-500 ease-out">
                <h1 className="font-heading font-extrabold text-[clamp(2rem,4vw,3rem)] leading-[1.1] pb-[0.08em] mb-2 tracking-tight">
                    Contribute a <em className="text-brand not-italic">Sign</em>
                </h1>
                <p className="text-muted leading-relaxed max-w-[500px]">
                    Add a new word to SentiSign's vocabulary. The system checks for duplicates automatically before recording starts.
                </p>
            </div>

            <WizardLayout steps={STEPS} currentStep={currentStep}>
                <AnimatePresence mode="wait">
                    {currentStep === 1 && (
                        <Gate1WordCheck
                            key="gate1"
                            onWordChecked={handleWordChecked}
                            onAddMoreSamples={handleAddMoreSamples}
                        />
                    )}

                    {currentStep === 2 && (
                        <Gate2Gesture
                            key="gate2"
                            word={targetWord}
                            onBack={handleGate2Back}
                            onContinue={handleGate2Continue}
                        />
                    )}

                    {currentStep === 3 && (
                        <Gate3Record
                            key="gate3"
                            word={targetWord}
                            onBack={handleGate3Back}
                            onSubmit={handleGate3Submit}
                        />
                    )}

                    {currentStep === 4 && (
                        <Gate4Success
                            key="gate4"
                            word={targetWord}
                            onReset={handleReset}
                        />
                    )}
                </AnimatePresence>
            </WizardLayout>
        </div>
    );
}
