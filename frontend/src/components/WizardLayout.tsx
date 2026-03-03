import type { ReactNode } from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface StepDescriptor {
    num: number;
    label: string;
}

interface WizardLayoutProps {
    steps: StepDescriptor[];
    currentStep: number;
    children: ReactNode;
}

export function WizardLayout({ steps, currentStep, children }: WizardLayoutProps) {
    return (
        <div className="flex flex-col gap-10 w-full animate-in fade-in zoom-in-95 duration-500 ease-out">

            {/* STEPS INDICATOR */}
            <div className="flex bg-surface border border-border-color rounded-xl overflow-hidden shadow-sm">
                {steps.map((step) => {
                    const isActive = currentStep === step.num;
                    const isDone = currentStep > step.num;

                    return (
                        <div
                            key={step.num}
                            className={twMerge(
                                clsx(
                                    "flex-1 text-center py-3.5 px-4 transition-all duration-300 border-r border-border-color last:border-r-0 relative",
                                    isActive ? "bg-[rgba(51,153,255,0.08)] before:absolute before:bottom-0 before:left-0 before:right-0 before:h-0.5 before:bg-brand" : "",
                                    isDone ? "bg-[rgba(51,153,255,0.03)]" : ""
                                )
                            )}
                        >
                            <div
                                className={twMerge(
                                    clsx(
                                        "font-heading font-extrabold text-[1.2rem]",
                                        isActive || isDone ? "text-brand" : "text-muted"
                                    )
                                )}
                            >
                                {step.num}
                            </div>
                            <div
                                className={twMerge(
                                    clsx(
                                        "text-[0.75rem] font-medium mt-0.5 transition-colors",
                                        isActive ? "text-text" : "text-muted"
                                    )
                                )}
                            >
                                {step.label}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* CONTENT AREA */}
            <div className="relative">
                {children}
            </div>

        </div>
    );
}
