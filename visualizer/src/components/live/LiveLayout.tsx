'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { ObservabilityPanel } from '@/components/live/ObservabilityPanel';
import { TerminalInputPanel } from '@/components/live/TerminalInputPanel';
import { TrajectoryGraphPanel } from '@/components/live/TrajectoryGraphPanel';
import {
  DEFAULT_OBSERVABILITY,
  LiveEvent,
  ObservabilityState,
  TrajectoryNode,
  applyMetricsEvent,
  applyNodeEvent,
} from '@/lib/live-types';

const STREAM_BASE = process.env.NEXT_PUBLIC_RLM_STREAM_URL ?? 'http://127.0.0.1:8765';
const DEFAULT_BACKEND = process.env.NEXT_PUBLIC_RLM_DEFAULT_BACKEND ?? 'openai';

const LISTENED_EVENT_NAMES = ['node_start', 'node_complete', 'error', 'metrics', 'iteration', 'run_complete'];

function parseLiveEvent(raw: MessageEvent<string>): LiveEvent | null {
  try {
    const parsed = JSON.parse(raw.data) as LiveEvent;
    if (!parsed || typeof parsed !== 'object' || typeof parsed.event_type !== 'string') {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function LiveLayout() {
  const [backend, setBackend] = useState(DEFAULT_BACKEND);
  const [prompt, setPrompt] = useState(
    'Use rlm_query to evaluate two alternative plans for reducing flaky tests in a CI pipeline, then return your recommendation with FINAL_VAR.',
  );
  const [modelName, setModelName] = useState('gpt-5-nano');
  const [runId, setRunId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [nodes, setNodes] = useState<Record<string, TrajectoryNode>>({});
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [metrics, setMetrics] = useState<ObservabilityState>(DEFAULT_OBSERVABILITY);
  const [collapsedNodeIds, setCollapsedNodeIds] = useState<Set<string>>(new Set());
  const [connectionState, setConnectionState] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>(
    'disconnected',
  );
  const [runStartTime, setRunStartTime] = useState<number | null>(null);

  const [sidebarWidth, setSidebarWidth] = useState(320);
  const isDragging = useRef(false);

  const startDrag = useCallback((e: React.PointerEvent) => {
    isDragging.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, []);

  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging.current) return;
      
      const newWidth = e.clientX;
      if (newWidth < 250) setSidebarWidth(250);
      else if (newWidth > Math.max(300, window.innerWidth * 0.5)) setSidebarWidth(Math.max(300, window.innerWidth * 0.5));
      else setSidebarWidth(newWidth);
    };

    const onPointerUp = () => {
      if (!isDragging.current) return;
      isDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, []);

  const resetRunState = useCallback(() => {
    setNodes({});
    setEvents([]);
    setMetrics(DEFAULT_OBSERVABILITY);
    setCollapsedNodeIds(new Set());
  }, []);

  const handleEvent = useCallback((event: LiveEvent) => {
    setEvents((previous) => {
      const next = [event, ...previous];
      return next.slice(0, 200);
    });
    setNodes((previous) => applyNodeEvent(previous, event));
    setMetrics((previous) => applyMetricsEvent(previous, event));

    if (event.event_type === 'error') {
      setIsRunning(false);
      setRunStartTime(null);
    }
    if (event.event_type === 'run_complete') {
      setIsRunning(false);
      setRunStartTime(null);
    }
  }, []);

  useEffect(() => {
    if (!runId) {
      return;
    }

    setConnectionState('connecting');
    const source = new EventSource(`${STREAM_BASE}/events?run_id=${encodeURIComponent(runId)}`);

    const onEvent = (raw: MessageEvent<string>) => {
      const parsed = parseLiveEvent(raw);
      if (!parsed) {
        return;
      }
      setConnectionState('connected');
      handleEvent(parsed);
    };

    source.onmessage = onEvent;
    for (const eventName of LISTENED_EVENT_NAMES) {
      source.addEventListener(eventName, (rawEvent) => {
        const parsed = parseLiveEvent(rawEvent as MessageEvent<string>);
        if (!parsed) {
          return;
        }
        setConnectionState('connected');
        handleEvent(parsed);
      });
    }

    source.onerror = () => {
      setConnectionState('error');
    };

    return () => {
      source.close();
    };
  }, [runId, handleEvent]);

  const runRlm = useCallback(async () => {
    resetRunState();
    setConnectionState('connecting');
    setIsRunning(true);
    setRunStartTime(Date.now());

    try {
      const response = await fetch(`${STREAM_BASE}/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          config: {
            backend,
            backend_kwargs: {
              model_name: modelName,
            },
            environment: 'local',
            max_depth: 3,
            max_iterations: 8,
          },
        }),
      });

      if (!response.ok) {
        const errorBody = (await response.json()) as { error?: string };
        throw new Error(errorBody.error ?? `Run request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as { run_id: string };
      setRunId(payload.run_id);
    } catch (error) {
      setIsRunning(false);
      setRunStartTime(null);
      setConnectionState('error');
      const message = error instanceof Error ? error.message : 'Unknown request error';
      setMetrics((previous) => ({
        ...previous,
        lastError: message,
        phase: 'request_error',
      }));
    }
  }, [backend, modelName, prompt, resetRunState]);

  const toggleNode = useCallback((nodeId: string) => {
    setCollapsedNodeIds((previous) => {
      const next = new Set(previous);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);

  return (
    <div className="h-screen flex flex-col bg-[#0a0e1a] text-slate-100 font-sans">
      {/* Header */}
      <header className="shrink-0 border-b border-slate-800/80 bg-[#0d1220]/90 backdrop-blur-sm">
        <div className="flex items-center justify-between px-5 py-3">
          <div className="flex items-center gap-4">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-cyan-400/80 font-medium">RLM DevTools</p>
              <h1 className="text-[15px] font-semibold text-slate-50 tracking-tight">Live Trajectory Stream</h1>
            </div>
            {isRunning && (
              <div className="flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-950/40 px-3 py-1">
                <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span className="text-[10px] text-cyan-300 font-mono font-medium">LIVE</span>
              </div>
            )}
          </div>
          <Link
            href="/"
            className="rounded border border-slate-700 px-3 py-1 font-mono text-xs text-slate-400 hover:border-cyan-400/60 hover:text-cyan-200 transition-colors"
          >
            Back To Logs
          </Link>
        </div>
      </header>

      <main className="flex-1 min-h-0 flex overflow-hidden">
        {/* Left sidebar: Terminal + Observability stacked */}
        <div style={{ width: sidebarWidth }} className="shrink-0 flex flex-col h-full border-r border-slate-800/60 bg-[#0c1018] overflow-hidden">
          <div className="flex-1 min-h-0 overflow-y-auto w-full">
            <TerminalInputPanel
              prompt={prompt}
              backend={backend}
              modelName={modelName}
              onBackendChange={setBackend}
              onPromptChange={setPrompt}
              onModelNameChange={setModelName}
              onRun={runRlm}
              isRunning={isRunning}
              runId={runId}
              streamBase={STREAM_BASE}
              connectionState={connectionState}
              recentEvents={events}
            />

            <ObservabilityPanel
              metrics={metrics}
              eventCount={events.length}
              nodeCount={Object.keys(nodes).length}
              isRunning={isRunning}
              runStartTime={runStartTime}
            />
          </div>
        </div>

        {/* Resize handle */}
        <div 
          onPointerDown={startDrag}
          className="w-1.5 shrink-0 bg-slate-900 border-x border-slate-800/80 hover:bg-cyan-800/40 active:bg-cyan-700/50 transition-colors cursor-col-resize z-10" 
        />

        {/* Right: Full-width trajectory graph */}
        <div className="flex-1 min-w-0 h-full">
          <TrajectoryGraphPanel nodes={nodes} collapsedNodeIds={collapsedNodeIds} onToggleNode={toggleNode} />
        </div>
      </main>
    </div>
  );
}
