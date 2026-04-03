export type NodeStatus = 'idle' | 'running' | 'complete' | 'error';

export interface LiveEvent {
  event_id?: number;
  run_id: string;
  event_type: string;
  timestamp: string;
  [key: string]: unknown;
}

export interface IterationEntry {
  iteration: number;
  promptPreview: string;
  responsePreview: string;
  codeBlockCount: number;
  timeMs: number | null;
}

export interface TrajectoryNode {
  id: string;
  parentId: string | null;
  depth: number;
  model: string;
  promptPreview: string;
  /** Latest / final response (set on node_complete or run_complete). */
  responsePreview: string;
  /** Accumulated per-iteration outputs for root-level nodes. */
  iterations: IterationEntry[];
  status: NodeStatus;
  durationMs?: number;
  error?: string;
  children: string[];
}

export interface ObservabilityState {
  tokensInput: number;
  tokensOutput: number;
  latencyMs: number;
  recursionDepth: number;
  activeNodeId: string | null;
  iteration: number | null;
  phase: string;
  lastError: string | null;
  finalResponse: string | null;
}

export const DEFAULT_OBSERVABILITY: ObservabilityState = {
  tokensInput: 0,
  tokensOutput: 0,
  latencyMs: 0,
  recursionDepth: 0,
  activeNodeId: null,
  iteration: null,
  phase: 'idle',
  lastError: null,
  finalResponse: null,
};

function asString(value: unknown, fallback = ''): string {
  if (typeof value === 'string') {
    return value;
  }
  return fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  return fallback;
}

export function applyNodeEvent(
  previous: Record<string, TrajectoryNode>,
  event: LiveEvent,
): Record<string, TrajectoryNode> {
  // ── iteration: ACCUMULATE into iterations array (don't overwrite) ──
  if (event.event_type === 'iteration') {
    const nodeId = asString(event.node_id);
    if (!nodeId) return previous;
    const existing = previous[nodeId];
    if (!existing) return previous;
    const preview = asString(event.response_preview);
    if (!preview) return previous;

    const iterNum = typeof event.iteration === 'number' ? event.iteration : (existing.iterations.length + 1);
    const codeBlockCount = typeof event.code_block_count === 'number' ? event.code_block_count : 0;
    const timeMs = typeof event.iteration_time_ms === 'number' ? event.iteration_time_ms : null;

    const newEntry: IterationEntry = {
      iteration: iterNum,
      promptPreview: asString(event.prompt_preview),
      responsePreview: preview,
      codeBlockCount,
      timeMs,
    };

    return {
      ...previous,
      [nodeId]: {
        ...existing,
        // Append to iterations — never overwrite
        iterations: [...existing.iterations, newEntry],
        // Also keep latest as responsePreview for convenience
        responsePreview: preview,
      },
    };
  }

  // ── run_complete: set root's final response ──
  if (event.event_type === 'run_complete') {
    const response = asString(event.response);
    if (!response) return previous;
    const rootNode = previous['root'];
    if (!rootNode) return previous;
    return {
      ...previous,
      root: { ...rootNode, responsePreview: response },
    };
  }

  // ── node_start / node_complete ──
  if (event.event_type !== 'node_start' && event.event_type !== 'node_complete') {
    return previous;
  }

  const nodeId = asString(event.node_id);
  if (!nodeId) {
    return previous;
  }

  const parentIdValue = event.parent_id;
  const parentId = typeof parentIdValue === 'string' ? parentIdValue : null;
  const existing = previous[nodeId];

  const nextNode: TrajectoryNode = {
    id: nodeId,
    parentId,
    depth: asNumber(event.depth, existing?.depth ?? 0),
    model: asString(event.model, existing?.model ?? 'unknown'),
    promptPreview: asString(event.prompt_preview, existing?.promptPreview ?? ''),
    responsePreview: asString(event.response_preview, existing?.responsePreview ?? ''),
    iterations: existing?.iterations ?? [],
    status:
      event.event_type === 'node_start'
        ? 'running'
        : event.error
          ? 'error'
          : 'complete',
    durationMs: asNumber(event.duration_ms, existing?.durationMs ?? 0),
    error: asString(event.error, existing?.error ?? ''),
    children: existing?.children ?? [],
  };

  const next: Record<string, TrajectoryNode> = {
    ...previous,
    [nodeId]: nextNode,
  };

  if (parentId) {
    const parent = next[parentId];
    if (parent) {
      const alreadyHasChild = parent.children.includes(nodeId);
      if (!alreadyHasChild) {
        next[parentId] = {
          ...parent,
          children: [...parent.children, nodeId],
        };
      }
    }
  }

  return next;
}

export function applyMetricsEvent(
  previous: ObservabilityState,
  event: LiveEvent,
): ObservabilityState {
  if (event.event_type !== 'metrics' && event.event_type !== 'error' && event.event_type !== 'run_complete') {
    return previous;
  }

  if (event.event_type === 'error') {
    return {
      ...previous,
      lastError: asString(event.message, 'Unknown error'),
      phase: 'error',
    };
  }

  if (event.event_type === 'run_complete') {
    return {
      ...previous,
      phase: asString(event.status, 'complete'),
      lastError: asString(event.error, previous.lastError ?? ''),
      finalResponse: asString(event.response, previous.finalResponse ?? ''),
    };
  }

  const next = { ...previous };

  const tokens = event.tokens;
  if (tokens && typeof tokens === 'object' && !Array.isArray(tokens)) {
    const tokenRecord = tokens as Record<string, unknown>;
    next.tokensInput = asNumber(tokenRecord.input, previous.tokensInput);
    next.tokensOutput = asNumber(tokenRecord.output, previous.tokensOutput);
  }

  next.latencyMs = asNumber(event.latency_ms, previous.latencyMs);
  next.recursionDepth = asNumber(event.recursion_depth, previous.recursionDepth);

  const iterationValue = event.iteration;
  if (typeof iterationValue === 'number') {
    next.iteration = iterationValue;
  }

  const activeNodeId = event.active_node_id;
  if (typeof activeNodeId === 'string') {
    next.activeNodeId = activeNodeId;
  }

  next.phase = asString(event.phase, previous.phase);

  return next;
}

export function getRootNodes(nodes: Record<string, TrajectoryNode>): TrajectoryNode[] {
  return Object.values(nodes)
    .filter((node) => node.parentId === null)
    .sort((a, b) => a.depth - b.depth || a.id.localeCompare(b.id));
}
