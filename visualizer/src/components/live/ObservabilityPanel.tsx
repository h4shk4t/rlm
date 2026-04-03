'use client';

import { useEffect, useState } from 'react';

import { ObservabilityState } from '@/lib/live-types';

interface ObservabilityPanelProps {
  metrics: ObservabilityState;
  eventCount: number;
  nodeCount: number;
  isRunning: boolean;
  runStartTime: number | null;
}

function MetricRow({ label, value, accent }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1 text-[12px]">
      <span className="text-slate-500">{label}</span>
      <span className={accent ? 'text-cyan-300 font-semibold font-mono' : 'text-slate-200 font-mono'}>{value}</span>
    </div>
  );
}

function ElapsedTimer({ startTime }: { startTime: number }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime]);

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return (
    <span className="tabular-nums">
      {mins > 0 ? `${mins}m ${secs}s` : `${secs}s`}
    </span>
  );
}

function FinalResponseBlock({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = text.length > 300;
  const display = expanded || !isLong ? text : text.slice(0, 300) + '…';

  return (
    <div className="rounded-md border border-emerald-800/50 bg-emerald-950/20 p-3">
      <p className="mb-1.5 text-[11px] text-emerald-400/80 font-semibold">Final Response</p>
      <pre className="whitespace-pre-wrap break-words text-[12px] leading-relaxed text-emerald-100/90">
        {display}
      </pre>
      {isLong && (
        <button
          className="mt-1.5 text-[11px] text-cyan-400 hover:text-cyan-200 transition-colors"
          onClick={() => setExpanded((prev) => !prev)}
          type="button"
        >
          {expanded ? '▴ Collapse' : '▾ Show full response'}
        </button>
      )}
    </div>
  );
}

const PHASE_LABELS: Record<string, { color: string; label: string }> = {
  idle: { color: 'bg-slate-500', label: 'Idle' },
  run_start: { color: 'bg-cyan-400', label: 'Starting' },
  subcall_start: { color: 'bg-cyan-400 animate-pulse', label: 'Sub-call' },
  subcall_complete: { color: 'bg-emerald-400', label: 'Sub-call Done' },
  iteration_start: { color: 'bg-violet-400 animate-pulse', label: 'Iterating' },
  iteration_complete: { color: 'bg-violet-400', label: 'Iter Done' },
  run_complete: { color: 'bg-emerald-400', label: 'Complete' },
  success: { color: 'bg-emerald-400', label: 'Success' },
  error: { color: 'bg-rose-400', label: 'Error' },
  request_error: { color: 'bg-rose-400', label: 'Request Error' },
};

export function ObservabilityPanel({ metrics, eventCount, nodeCount, isRunning, runStartTime }: ObservabilityPanelProps) {
  const phaseCfg = PHASE_LABELS[metrics.phase] ?? { color: 'bg-slate-500', label: metrics.phase };

  return (
    <div className="p-4 space-y-3">
      <h2 className="text-[13px] font-semibold text-slate-200 tracking-tight">Observability</h2>

      {/* Status */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-3 space-y-2">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${phaseCfg.color}`} />
          <span className="text-[12px] text-slate-200">{phaseCfg.label}</span>
          {isRunning && runStartTime && (
            <span className="text-[12px] text-cyan-300 font-mono ml-auto">
              <ElapsedTimer startTime={runStartTime} />
            </span>
          )}
        </div>
      </div>

      {/* Core Metrics */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-3 divide-y divide-slate-800/50">
        <MetricRow label="Active Node" value={metrics.activeNodeId ?? 'n/a'} accent />
        <MetricRow label="Depth" value={metrics.recursionDepth} />
        <MetricRow label="Iteration" value={metrics.iteration ?? 'n/a'} />
        <MetricRow
          label="Latency"
          value={
            metrics.latencyMs > 1000
              ? `${(metrics.latencyMs / 1000).toFixed(1)}s`
              : `${metrics.latencyMs}ms`
          }
        />
      </div>

      {/* Tokens */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-3">
        <p className="text-[11px] text-slate-500 mb-2">Tokens</p>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded bg-slate-800/50 p-2 text-center">
            <p className="text-[10px] text-slate-500 uppercase">Input</p>
            <p className="text-[14px] font-mono font-semibold text-slate-100 tabular-nums">
              {metrics.tokensInput.toLocaleString()}
            </p>
          </div>
          <div className="rounded bg-slate-800/50 p-2 text-center">
            <p className="text-[10px] text-slate-500 uppercase">Output</p>
            <p className="text-[14px] font-mono font-semibold text-slate-100 tabular-nums">
              {metrics.tokensOutput.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Counts */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-3 divide-y divide-slate-800/50">
        <MetricRow label="Events" value={eventCount} />
        <MetricRow label="Nodes" value={nodeCount} />
      </div>

      {/* Error */}
      {metrics.lastError && (
        <div className="rounded-md border border-rose-800/50 bg-rose-950/20 p-3">
          <p className="text-[11px] text-rose-400/80 font-semibold mb-1">Error</p>
          <p className="text-[12px] text-rose-200/90 break-words">{metrics.lastError}</p>
        </div>
      )}

      {/* Final Response */}
      {metrics.finalResponse && <FinalResponseBlock text={metrics.finalResponse} />}
    </div>
  );
}
