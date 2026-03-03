import type { ReactNode } from 'react';
import { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Loader2, ServerOff, TriangleAlert, XCircle } from 'lucide-react';

type BackendStepState = 'pending' | 'loading' | 'done' | 'error';
type BackendOverallState = 'starting' | 'loading' | 'ready' | 'error';

type BackendStep = {
    id: string;
    label: string;
    state: BackendStepState;
    detail: string | null;
};

type BackendStatus = {
    state: BackendOverallState;
    message: string;
    updated_at?: string;
    core_ready: boolean;
    ready: boolean;
    steps: BackendStep[];
};

function isRecord(v: unknown): v is Record<string, unknown> {
    return typeof v === 'object' && v !== null;
}

function parseBackendStatus(raw: unknown): BackendStatus | null {
    if (!isRecord(raw)) return null;

    const state = raw.state;
    const message = raw.message;
    const coreReady = raw.core_ready;
    const ready = raw.ready;
    const stepsRaw = raw.steps;

    const okState = state === 'starting' || state === 'loading' || state === 'ready' || state === 'error';
    if (!okState) return null;
    if (typeof message !== 'string') return null;
    if (typeof coreReady !== 'boolean') return null;
    if (typeof ready !== 'boolean') return null;
    if (!Array.isArray(stepsRaw)) return null;

    const steps: BackendStep[] = [];
    for (const s of stepsRaw) {
        if (!isRecord(s)) continue;
        const id = s.id;
        const label = s.label;
        const st = s.state;
        const detail = s.detail;
        const okStepState = st === 'pending' || st === 'loading' || st === 'done' || st === 'error';
        if (typeof id !== 'string' || typeof label !== 'string' || !okStepState) continue;
        steps.push({
            id,
            label,
            state: st,
            detail: typeof detail === 'string' ? detail : null,
        });
    }

    return {
        state,
        message,
        updated_at: typeof raw.updated_at === 'string' ? raw.updated_at : undefined,
        core_ready: coreReady,
        ready,
        steps,
    };
}

type GateState =
    | { kind: 'checking' }
    | { kind: 'offline' }
    | { kind: 'loading'; status: BackendStatus }
    | { kind: 'ready'; status: BackendStatus }
    | { kind: 'error'; message: string };

export function BackendGate({ children }: { children: ReactNode }) {
    const [gate, setGate] = useState<GateState>({ kind: 'checking' });
    const statusUrl = import.meta.env.DEV ? 'http://127.0.0.1:8000/api/status' : '/api/status';

    useEffect(() => {
        let ignore = false;
        let timer: number | null = null;
        let delayMs = 900;
        let stopped = false;

        async function poll() {
            const controller = new AbortController();
            const timeoutId = window.setTimeout(() => controller.abort(), 1200);
            try {
                const res = await fetch(statusUrl, { cache: 'no-store', signal: controller.signal });
                const json = await res.json().catch(() => null);

                if (ignore) return;
                if (!res.ok) {
                    setGate({ kind: 'error', message: `Backend returned ${res.status}` });
                    delayMs = 2000;
                    return;
                }

                const status = parseBackendStatus(json);
                if (!status) {
                    setGate({ kind: 'error', message: 'Backend status payload is invalid.' });
                    delayMs = 2000;
                    return;
                }

                if (status.ready) {
                    setGate({ kind: 'ready', status });
                    stopped = true;
                    return;
                }
                setGate({ kind: 'loading', status });
                delayMs = 800;
            } catch (e) {
                if (ignore) return;
                // Most common case: backend not started yet.
                setGate({ kind: 'offline' });
                delayMs = Math.min(8000, Math.max(1500, Math.floor(delayMs * 1.5)));
            } finally {
                window.clearTimeout(timeoutId);
                if (ignore) return;
                if (!stopped) timer = window.setTimeout(poll, delayMs);
            }
        }

        poll();

        return () => {
            ignore = true;
            if (timer) window.clearTimeout(timer);
            stopped = true;
        };
    }, []);

    const body = useMemo(() => {
        if (gate.kind === 'checking') {
            return {
                title: 'Connecting to backend',
                message: 'Waiting for /api/status...',
                icon: <Loader2 className="w-5 h-5 animate-spin text-brand" />,
                steps: null as BackendStatus['steps'] | null,
            };
        }
        if (gate.kind === 'offline') {
            return {
                title: 'Backend offline',
                message: 'Frontend is up, but the API server is not reachable.',
                icon: <ServerOff className="w-5 h-5 text-amber" />,
                steps: null as BackendStatus['steps'] | null,
            };
        }
        if (gate.kind === 'error') {
            return {
                title: 'Backend error',
                message: gate.message,
                icon: <TriangleAlert className="w-5 h-5 text-amber" />,
                steps: null as BackendStatus['steps'] | null,
            };
        }
        if (gate.kind === 'loading') {
            return {
                title: 'Backend is loading',
                message: gate.status.message,
                icon: <Loader2 className="w-5 h-5 animate-spin text-brand" />,
                steps: gate.status.steps,
            };
        }
        return null;
    }, [gate]);

    if (gate.kind === 'ready') {
        return <>{children}</>;
    }

    const steps = body?.steps ?? null;
    const doneCount = steps ? steps.filter((s) => s.state === 'done').length : 0;
    const totalCount = steps ? steps.length : 0;
    const pct = totalCount ? Math.round((doneCount / totalCount) * 100) : 0;

    return (
        <div className="fixed inset-0 z-[9999] bg-bg">
            <div className="absolute inset-0 opacity-60 pointer-events-none"
                style={{
                    background:
                        'radial-gradient(900px 500px at 20% 15%, rgba(51,153,255,0.24), transparent 60%),' +
                        'radial-gradient(700px 420px at 85% 80%, rgba(255,127,64,0.18), transparent 60%)'
                }}
            />

            <div className="relative max-w-[740px] mx-auto px-6 py-14">
                <div className="bg-surface border border-border-color rounded-2xl p-7 shadow-[0_18px_52px_rgba(15,34,68,0.14)]">
                    <div className="flex items-center gap-3 mb-3">
                        {body?.icon}
                        <h1 className="font-heading font-extrabold text-[1.35rem] tracking-tight">
                            {body?.title}
                        </h1>
                    </div>

                    <p className="text-muted leading-relaxed mb-5">
                        {body?.message}
                    </p>

                    {steps && steps.length > 0 && (
                        <>
                            <div className="mb-4">
                                <div className="flex justify-between text-[0.8rem] text-muted mb-2">
                                    <span>Startup progress</span>
                                    <span className="font-mono">{pct}%</span>
                                </div>
                                <div className="h-2 bg-border-color rounded-full overflow-hidden">
                                    <div className="h-full bg-brand transition-all duration-300" style={{ width: `${pct}%` }} />
                                </div>
                            </div>

                            <div className="grid grid-cols-1 gap-2">
                                {steps.map((s) => (
                                    <div
                                        key={s.id}
                                        className="flex items-start gap-3 bg-bg/40 border border-border-color rounded-xl px-4 py-3"
                                    >
                                        <div className="mt-[2px]">
                                            {s.state === 'done' && <CheckCircle2 className="w-4.5 h-4.5 text-brand" />}
                                            {s.state === 'loading' && <Loader2 className="w-4.5 h-4.5 animate-spin text-brand" />}
                                            {s.state === 'error' && <XCircle className="w-4.5 h-4.5 text-red" />}
                                            {s.state === 'pending' && <div className="w-2.5 h-2.5 rounded-full bg-muted mt-1" />}
                                        </div>
                                        <div className="flex-1">
                                            <div className="text-[0.95rem] font-medium">{s.label}</div>
                                            {s.state === 'error' && s.detail && (
                                                <div className="text-[0.8rem] text-red mt-1 break-words">{s.detail}</div>
                                            )}
                                        </div>
                                        <div className="text-[0.75rem] font-mono text-muted uppercase tracking-wider mt-[2px]">
                                            {s.state}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    {(gate.kind === 'offline' || gate.kind === 'error') && (
                        <div className="mt-6 text-[0.85rem] text-muted">
                            Start the backend (`make backend`) then this page will continue automatically.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
