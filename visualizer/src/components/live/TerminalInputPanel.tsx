'use client';

import { LiveEvent } from '@/lib/live-types';
import { Button } from '@/components/ui/button';
import { useRef, useEffect } from 'react';

interface TerminalInputPanelProps {
  prompt: string;
  backend: string;
  modelName: string;
  onBackendChange: (value: string) => void;
  onPromptChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onRun: () => Promise<void>;
  isRunning: boolean;
  runId: string | null;
  streamBase: string;
  connectionState: 'disconnected' | 'connecting' | 'connected' | 'error';
  recentEvents: LiveEvent[];
}

function connectionBadgeClass(state: TerminalInputPanelProps['connectionState']): string {
  if (state === 'connected') return 'bg-emerald-500/25 text-emerald-200 border-emerald-500/40';
  if (state === 'connecting') return 'bg-amber-500/20 text-amber-100 border-amber-500/40';
  if (state === 'error') return 'bg-rose-500/20 text-rose-100 border-rose-500/40';
  return 'bg-slate-700/40 text-slate-300 border-slate-600/60';
}

const EVENT_TYPE_COLORS: Record<string, string> = {
  node_start: 'text-cyan-400',
  node_complete: 'text-emerald-400',
  iteration: 'text-violet-400',
  metrics: 'text-slate-500',
  error: 'text-rose-400',
  run_complete: 'text-amber-400',
};

function EventLine({ event }: { event: LiveEvent }) {
  const color = EVENT_TYPE_COLORS[event.event_type] ?? 'text-slate-400';
  const nodeId = typeof event.node_id === 'string' ? event.node_id : '';
  const dur = typeof event.duration_ms === 'number' ? ` ${event.duration_ms}ms` : '';

  let detail = '';
  if (event.event_type === 'node_complete' || event.event_type === 'iteration') {
    const preview = typeof event.response_preview === 'string' ? event.response_preview.slice(0, 60) : '';
    if (preview) detail = preview;
  }
  if (event.event_type === 'error') {
    detail = typeof event.message === 'string' ? event.message.slice(0, 60) : '';
  }

  return (
    <div className="py-0.5 border-b border-slate-800/40 last:border-0">
      <div className="flex items-center gap-1.5">
        <span className={`shrink-0 text-[11px] font-medium ${color}`}>
          {event.event_type}
        </span>
        {nodeId && <span className="text-slate-500 text-[10px]">{nodeId}</span>}
        {dur && <span className="text-slate-600 text-[10px] ml-auto tabular-nums">{dur}</span>}
      </div>
      {detail && (
        <p className="text-[10px] text-slate-500/80 truncate mt-0.5 pl-2">{detail}</p>
      )}
    </div>
  );
}

export function TerminalInputPanel({
  prompt,
  backend,
  modelName,
  onBackendChange,
  onPromptChange,
  onModelNameChange,
  onRun,
  isRunning,
  runId,
  streamBase,
  connectionState,
  recentEvents,
}: TerminalInputPanelProps) {
  return (
    <div className="p-4 space-y-3 border-b border-slate-800/50">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-[13px] font-semibold text-slate-200 tracking-tight">Run Terminal</h2>
        <span className={`rounded border px-2 py-0.5 text-[10px] font-mono ${connectionBadgeClass(connectionState)}`}>
          {connectionState}
        </span>
      </div>

      {/* Config row */}
      <div className="grid grid-cols-2 gap-2">
        <label className="block space-y-1">
          <span className="text-[11px] text-slate-400">Backend</span>
          <select
            value={backend}
            onChange={(event) => onBackendChange(event.target.value)}
            className="w-full rounded-md border border-slate-700/80 bg-slate-900/90 px-2.5 py-1.5 text-[13px] text-slate-100 outline-none focus:border-cyan-400/70 transition-colors"
          >
            <option value="openai">openai</option>
            <option value="azure_openai">azure_openai</option>
          </select>
        </label>

        <label className="block space-y-1">
          <span className="text-[11px] text-slate-400">Model</span>
          <input
            value={modelName}
            onChange={(event) => onModelNameChange(event.target.value)}
            className="w-full rounded-md border border-slate-700/80 bg-slate-900/90 px-2.5 py-1.5 text-[13px] text-slate-100 outline-none focus:border-cyan-400/70 transition-colors"
            placeholder="gpt-5-nano"
          />
        </label>
      </div>

      {/* Prompt — taller textarea */}
      <label className="block space-y-1">
        <span className="text-[11px] text-slate-400">Prompt</span>
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          className="h-36 w-full resize-none rounded-md border border-slate-700/80 bg-slate-900/90 px-3 py-2 text-[13px] leading-relaxed text-slate-100 outline-none focus:border-cyan-400/70 transition-colors"
          placeholder="Describe the run task..."
        />
      </label>

      <Button
        className="w-full bg-cyan-600 text-cyan-50 hover:bg-cyan-500 transition-colors text-[13px] h-10"
        onClick={() => { void onRun(); }}
        disabled={isRunning || prompt.trim().length === 0}
      >
        {isRunning ? (
          <span className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-200 animate-pulse" />
            Running…
          </span>
        ) : (
          'Run RLM'
        )}
      </Button>

      {/* Connection info */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-2.5 text-[11px] text-slate-500 space-y-0.5">
        <p>run_id: <span className="text-slate-300">{runId ?? 'n/a'}</span></p>
        <p>backend: <span className="text-slate-300">{backend}</span></p>
        <p className="truncate">sse: <span className="text-slate-300">{streamBase}/events</span></p>
      </div>

      {/* Events feed */}
      <div className="rounded-md border border-slate-800/80 bg-[#080c14] p-2.5">
        <p className="mb-1.5 text-[11px] text-slate-500">
          Events <span className="text-slate-600">({recentEvents.length})</span>
        </p>
        <div className="max-h-32 overflow-y-auto space-y-0">
          {recentEvents.length === 0 ? (
            <p className="text-[11px] text-slate-600">No events yet.</p>
          ) : (
            recentEvents.slice(0, 20).map((event, index) => (
              <EventLine key={`${event.event_id ?? index}`} event={event} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}
