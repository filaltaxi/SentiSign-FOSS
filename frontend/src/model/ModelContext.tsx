import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

export type SignModel = 'mlp' | 'lstm';

const STORAGE_KEY = 'sentisign:model';

const MODEL_LABELS: Record<SignModel, string> = {
    mlp: 'Landmark MLP',
    lstm: 'Temporal LSTM',
};

type ModelAvailability = Record<SignModel, boolean>;

type ModelContextValue = {
    model: SignModel | null;
    availableModels: ModelAvailability;
    setModel: (next: SignModel) => void;
    isOnboardingOpen: boolean;
    sessionResetNonce: number;
};

const ModelContext = createContext<ModelContextValue | null>(null);

type ModelsResponse = {
    models?: Array<{ id?: string; available?: boolean }>;
};

function parseStoredModel(raw: string | null): SignModel | null {
    if (raw === 'mlp' || raw === 'lstm') return raw;
    return null;
}

function deriveAvailability(payload: ModelsResponse | null): ModelAvailability {
    const defaults: ModelAvailability = { mlp: true, lstm: false };
    if (!payload || !Array.isArray(payload.models)) return defaults;

    for (const model of payload.models) {
        if (model?.id === 'mlp') defaults.mlp = model.available !== false;
        if (model?.id === 'lstm') defaults.lstm = model.available === true;
    }

    defaults.mlp = true;
    return defaults;
}

export function ModelProvider({ children }: { children: ReactNode }) {
    const [model, setModelState] = useState<SignModel | null>(null);
    const [availableModels, setAvailableModels] = useState<ModelAvailability>({ mlp: true, lstm: false });
    const [sessionResetNonce, setSessionResetNonce] = useState(0);

    useEffect(() => {
        const stored = parseStoredModel(window.localStorage.getItem(STORAGE_KEY));
        setModelState(stored);
    }, []);

    useEffect(() => {
        let ignore = false;
        const controller = new AbortController();

        (async () => {
            try {
                const response = await fetch('/api/models', { signal: controller.signal });
                const data = (await response.json()) as ModelsResponse;
                if (ignore) return;
                const availability = deriveAvailability(data);
                setAvailableModels(availability);

                setModelState((current) => {
                    const next = current ?? parseStoredModel(window.localStorage.getItem(STORAGE_KEY));
                    return next;
                });
            } catch {
                if (ignore) return;
                setAvailableModels({ mlp: true, lstm: false });
                setModelState((current) => current ?? parseStoredModel(window.localStorage.getItem(STORAGE_KEY)));
            }
        })();

        return () => {
            ignore = true;
            controller.abort();
        };
    }, []);

    const setModel = useCallback((next: SignModel) => {
        setModelState((prev) => {
            window.localStorage.setItem(STORAGE_KEY, next);
            if (prev !== null && prev !== next) {
                setSessionResetNonce((value) => value + 1);
            }
            return next;
        });
    }, []);

    const contextValue = useMemo<ModelContextValue>(() => {
        return {
            model,
            availableModels,
            setModel,
            isOnboardingOpen: model === null,
            sessionResetNonce,
        };
    }, [availableModels, model, sessionResetNonce, setModel]);

    return (
        <ModelContext.Provider value={contextValue}>
            {children}
            <ModelPickerModal
                isOpen={model === null}
                selected={model ?? 'mlp'}
                availableModels={availableModels}
                onSelect={setModel}
            />
        </ModelContext.Provider>
    );
}

function ModelPickerModal({
    isOpen,
    selected,
    availableModels,
    onSelect,
}: {
    isOpen: boolean;
    selected: SignModel;
    availableModels: ModelAvailability;
    onSelect: (next: SignModel) => void;
}) {
    const [draft, setDraft] = useState<SignModel>(selected);

    useEffect(() => {
        setDraft(selected);
    }, [selected]);

    if (!isOpen) return null;

    const commit = () => {
        onSelect(draft);
    };

    return (
        <div className="fixed inset-0 z-[10000] grid place-items-center bg-[rgba(8,18,32,0.56)] px-4">
            <div className="w-full max-w-[620px] rounded-3xl border border-border-color bg-white p-6 shadow-[0_26px_56px_rgba(8,18,32,0.35)] sm:p-7">
                <h2 className="font-heading text-[1.5rem] font-extrabold tracking-tight text-text">Choose your sign model</h2>
                <p className="mt-2 text-[0.96rem] text-muted">
                    You can switch this anytime from the top navbar.
                </p>

                <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <button
                        type="button"
                        onClick={() => setDraft('mlp')}
                        className={`rounded-2xl border px-4 py-4 text-left transition-all duration-200 ${draft === 'mlp'
                            ? 'border-[#9fc9ff] bg-[#edf5ff] shadow-[0_14px_26px_rgba(0,127,255,0.16)]'
                            : 'border-border-color bg-white hover:border-[#b8d4ff]'
                            }`}
                    >
                        <div className="text-[0.78rem] font-bold uppercase tracking-[0.14em] text-brand">MLP</div>
                        <div className="mt-1 font-heading text-[1.02rem] font-extrabold text-text">{MODEL_LABELS.mlp}</div>
                        <p className="mt-1.5 text-[0.84rem] text-muted">Fast per-frame predictions with the current landmark pipeline.</p>
                    </button>

                    <button
                        type="button"
                        onClick={() => setDraft('lstm')}
                        className={`rounded-2xl border px-4 py-4 text-left transition-all duration-200 ${draft === 'lstm'
                            ? 'border-[#ffc7a9] bg-[#fff1e8] shadow-[0_14px_26px_rgba(200,90,33,0.14)]'
                            : 'border-border-color bg-white hover:border-[#ffd3bf]'
                            }`}
                    >
                        <div className="text-[0.78rem] font-bold uppercase tracking-[0.14em] text-[#c85a21]">LSTM</div>
                        <div className="mt-1 font-heading text-[1.02rem] font-extrabold text-text">{MODEL_LABELS.lstm}</div>
                        <p className="mt-1.5 text-[0.84rem] text-muted">Motion-aware 60-frame recognition for temporal signing.</p>
                        {!availableModels.lstm && (
                            <p className="mt-2 text-[0.76rem] font-semibold text-[#b35b2f]">Model not trained yet: use this mode to contribute and train first.</p>
                        )}
                    </button>
                </div>

                <button
                    type="button"
                    onClick={commit}
                    className="mt-6 h-11 w-full rounded-xl bg-gradient-to-r from-brand to-brand-end text-[0.9rem] font-bold text-white shadow-[0_14px_24px_rgba(0,127,255,0.24)] transition-all duration-200 hover:brightness-105"
                >
                    Continue
                </button>
            </div>
        </div>
    );
}

export function useModel() {
    const ctx = useContext(ModelContext);
    if (!ctx) {
        throw new Error('useModel must be used within ModelProvider');
    }
    return ctx;
}
