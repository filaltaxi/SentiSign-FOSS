export type HandPoint = { x: number; y: number; z: number };

export type HandResults = {
    multiHandLandmarks?: HandPoint[][];
    multiHandedness?: Array<{ classification?: Array<{ label?: string }> }>;
};

const HAND_POINTS = 21;
const AXES = 3;
const HAND_DIM = HAND_POINTS * AXES;

function normalizeHand(points: HandPoint[]): number[] {
    if (!Array.isArray(points) || points.length !== HAND_POINTS) {
        return new Array(HAND_DIM).fill(0);
    }

    const wrist = points[0];
    const centered = points.map((point) => [
        Number(point.x) - Number(wrist.x),
        Number(point.y) - Number(wrist.y),
        Number(point.z) - Number(wrist.z),
    ]);

    let scale = 0;
    for (const [x, y, z] of centered) {
        scale = Math.max(scale, Math.abs(x), Math.abs(y), Math.abs(z));
    }

    const inv = scale > 0 ? 1 / scale : 1;
    const out: number[] = [];
    for (const [x, y, z] of centered) {
        out.push(x * inv, y * inv, z * inv);
    }
    return out;
}

function copyInto(target: number[], source: number[]) {
    for (let i = 0; i < HAND_DIM; i += 1) target[i] = source[i];
}

function wristX(points: HandPoint[] | undefined): number {
    const x = Number(points?.[0]?.x);
    return Number.isFinite(x) ? x : 0;
}

export function hasTemporalSignal(features: number[], eps = 1e-6): boolean {
    let sum = 0;
    for (let i = 0; i < features.length; i += 1) {
        sum += Math.abs(features[i]);
    }
    return sum > eps;
}

export function extractTemporalFeatures(results: HandResults): number[] {
    const right = new Array(HAND_DIM).fill(0);
    const left = new Array(HAND_DIM).fill(0);
    const landmarks = results.multiHandLandmarks ?? [];
    const normalizedHands = landmarks.map((hand) => normalizeHand(hand));
    const handedness = results.multiHandedness ?? [];
    let assigned = false;

    for (let i = 0; i < landmarks.length; i += 1) {
        if (i >= handedness.length) continue;
        const label = handedness[i]?.classification?.[0]?.label;
        const hand = normalizedHands[i];
        if (!hand || (label !== 'Right' && label !== 'Left')) continue;
        if (label === 'Right') copyInto(right, hand);
        else copyInto(left, hand);
        assigned = true;
    }

    if (!assigned && landmarks.length === 1) {
        const hand = normalizedHands[0];
        if (hand) {
            if (wristX(landmarks[0]) >= 0.5) copyInto(right, hand);
            else copyInto(left, hand);
        }
    }

    if (!assigned && landmarks.length >= 2) {
        const wrists = landmarks.map((hand, i) => ({ i, x: wristX(hand) }));
        wrists.sort((a, b) => a.x - b.x);
        const leftIndex = wrists[0]?.i ?? -1;
        const rightIndex = wrists[wrists.length - 1]?.i ?? -1;
        if (rightIndex >= 0) copyInto(right, normalizedHands[rightIndex]);
        if (leftIndex >= 0) copyInto(left, normalizedHands[leftIndex]);
    }

    return [...right, ...left];
}
