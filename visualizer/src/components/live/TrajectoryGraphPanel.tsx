'use client';

import { useEffect, useRef, useState } from 'react';

import { IterationEntry, TrajectoryNode, getRootNodes } from '@/lib/live-types';

interface TrajectoryGraphPanelProps {
  nodes: Record<string, TrajectoryNode>;
  collapsedNodeIds: Set<string>;
  onToggleNode: (nodeId: string) => void;
}

/* ── Status styling ────────────────────────────────────────────── */

const STATUS_CONFIG: Record<
  TrajectoryNode['status'],
  { border: string; bg: string; text: string; dot: string; dotAnimate: string }
> = {
  running: {
    border: 'border-cyan-400/50',
    bg: 'bg-[#0b1a2a]',
    text: 'text-cyan-100',
    dot: 'bg-cyan-400',
    dotAnimate: 'animate-pulse',
  },
  complete: {
    border: 'border-emerald-500/35',
    bg: 'bg-[#0b1a1a]',
    text: 'text-emerald-100',
    dot: 'bg-emerald-400',
    dotAnimate: '',
  },
  error: {
    border: 'border-rose-500/40',
    bg: 'bg-[#1a0b0f]',
    text: 'text-rose-100',
    dot: 'bg-rose-400',
    dotAnimate: '',
  },
  idle: {
    border: 'border-slate-700',
    bg: 'bg-[#0d1018]',
    text: 'text-slate-300',
    dot: 'bg-slate-500',
    dotAnimate: '',
  },
};

const DEPTH_COLORS = [
  'text-cyan-300',
  'text-violet-300',
  'text-amber-300',
  'text-rose-300',
  'text-teal-300',
  'text-fuchsia-300',
];

const DEPTH_BG = [
  'bg-cyan-500/[0.04]',
  'bg-violet-500/[0.04]',
  'bg-amber-500/[0.04]',
  'bg-rose-500/[0.04]',
  'bg-teal-500/[0.04]',
  'bg-fuchsia-500/[0.04]',
];

const DEPTH_ACCENT_BORDER = [
  'border-cyan-500/20',
  'border-violet-500/20',
  'border-amber-500/20',
  'border-rose-500/20',
  'border-teal-500/20',
  'border-fuchsia-500/20',
];

function depthColor(depth: number): string {
  return DEPTH_COLORS[depth % DEPTH_COLORS.length];
}

function depthBg(depth: number): string {
  return DEPTH_BG[depth % DEPTH_BG.length];
}

function depthAccent(depth: number): string {
  return DEPTH_ACCENT_BORDER[depth % DEPTH_ACCENT_BORDER.length];
}

/* ── Subcomponents ─────────────────────────────────────────────── */

function ExpandableText({
  text,
  previewLen = 500,
  className = '',
}: {
  text: string;
  previewLen?: number;
  className?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const isLong = text.length > previewLen;
  const display = expanded || !isLong ? text : text.slice(0, previewLen) + '…';

  return (
    <>
      <pre className={`whitespace-pre-wrap break-words ${className}`}>{display}</pre>
      {isLong && (
        <button
          className="mt-1 text-[11px] text-cyan-400 hover:text-cyan-200 transition-colors"
          onClick={() => setExpanded((prev) => !prev)}
          type="button"
        >
          {expanded ? '▴ Less' : '▾ Show full'}
        </button>
      )}
    </>
  );
}

/** Renders all accumulated iteration outputs for a node */
function IterationList({ iterations }: { iterations: IterationEntry[] }) {
  const [collapsed, setCollapsed] = useState(false);

  if (iterations.length === 0) return null;

  return (
    <div className="mt-3 space-y-2">
      <button
        className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition-colors font-medium"
        onClick={() => setCollapsed((prev) => !prev)}
        type="button"
      >
        <span>{collapsed ? '▸' : '▾'}</span>
        <span>Turns ({iterations.length})</span>
      </button>

      {!collapsed &&
        iterations.map((iter) => (
          <div
            key={iter.iteration}
            className="rounded-lg border border-slate-700/50 bg-[#080c16] overflow-hidden"
          >
            {/* Turn header */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/20 border-b border-slate-700/30">
              <span className="text-[11px] font-semibold text-violet-300/90">
                Turn {iter.iteration}
              </span>
              {iter.codeBlockCount > 0 && (
                <span className="text-[11px] text-slate-500">
                  {iter.codeBlockCount} code block{iter.codeBlockCount > 1 ? 's' : ''}
                </span>
              )}
              {iter.timeMs !== null && (
                <span className="text-[11px] text-slate-500 ml-auto tabular-nums font-mono">
                  {iter.timeMs > 1000
                    ? `${(iter.timeMs / 1000).toFixed(1)}s`
                    : `${iter.timeMs}ms`}
                </span>
              )}
            </div>

            {/* Input (prompt) */}
            {iter.promptPreview && (
              <div className="px-3 py-2 border-b border-slate-800/30">
                <p className="text-[11px] text-blue-400/70 font-medium mb-1">Input</p>
                <ExpandableText
                  text={iter.promptPreview}
                  previewLen={400}
                  className="text-[12px] leading-[1.65] text-blue-100/60"
                />
              </div>
            )}

            {/* Output (response) */}
            <div className="px-3 py-2">
              <p className="text-[11px] text-emerald-400/70 font-medium mb-1">Output</p>
              <ExpandableText
                text={iter.responsePreview}
                previewLen={600}
                className="text-[12px] leading-[1.65] text-emerald-100/80 max-h-52 overflow-y-auto"
              />
            </div>
          </div>
        ))}
    </div>
  );
}

/** Renders the final response (from node_complete) for child nodes */
function FinalOutputBlock({ text }: { text: string }) {
  return (
    <div className="mt-2 rounded-lg bg-[#080c16] px-3 py-2 border border-emerald-800/30">
      <p className="mb-1 text-[11px] text-emerald-400/70 font-medium">Output</p>
      <ExpandableText
        text={text}
        previewLen={500}
        className="text-[12px] leading-[1.65] text-emerald-100/85 max-h-60 overflow-y-auto"
      />
    </div>
  );
}

/* ── Node Card ───────────────────────────────────────────────── */

function NodeCard({
  node,
  nodes,
  isExpanded,
  onToggle,
}: {
  node: TrajectoryNode;
  nodes: Record<string, TrajectoryNode>;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const cfg = STATUS_CONFIG[node.status];
  const hasChildren = node.children.length > 0;
  const hasIterations = node.iterations.length > 0;

  return (
    <div
      className={`rounded-lg border ${cfg.border} ${cfg.bg} ${cfg.text} transition-all duration-200 shrink-0 ${
        isExpanded ? 'w-[520px]' : 'w-[300px]'
      }`}
    >
      {/* Card header */}
      <div
        className="flex items-center justify-between gap-3 px-3.5 py-2.5 cursor-pointer hover:brightness-125 transition-all select-none"
        onClick={onToggle}
      >
        <div className="flex min-w-0 items-center gap-2">
          {/* Status dot */}
          <div className={`h-2.5 w-2.5 rounded-full shrink-0 ${cfg.dot} ${cfg.dotAnimate}`} />

          {/* Node ID */}
          <span className="font-semibold text-[13px] text-slate-100 truncate">{node.id}</span>

          {/* Depth badge */}
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold bg-slate-800/70 ${depthColor(node.depth)}`}
          >
            d{node.depth}
          </span>

          {/* Model */}
          <span className="text-slate-500 truncate text-[11px]">{node.model}</span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {hasIterations && (
            <span className="text-[10px] text-violet-400/60 font-mono">
              {node.iterations.length} turns
            </span>
          )}

          {/* Duration */}
          {node.status === 'running' ? (
            <span className="text-cyan-400 animate-pulse text-[11px]">…</span>
          ) : node.durationMs ? (
            <span className="text-slate-400 text-[11px] tabular-nums font-mono">
              {node.durationMs > 1000
                ? `${(node.durationMs / 1000).toFixed(1)}s`
                : `${node.durationMs}ms`}
            </span>
          ) : null}

          {/* Expand indicator */}
          <span className="text-slate-500 text-[11px]">{isExpanded ? '▾' : '▸'}</span>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3.5 pb-3 border-t border-slate-800/30 pt-2.5 max-h-[600px] overflow-y-auto">
          {/* Prompt preview */}
          {node.promptPreview && (
            <div className="mb-2">
              <p className="text-[11px] text-blue-400/50 font-medium mb-1">Initial Prompt</p>
              <ExpandableText
                text={node.promptPreview}
                previewLen={300}
                className="text-[12px] text-blue-200/50 leading-[1.6]"
              />
            </div>
          )}

          {/* Children indicator */}
          {hasChildren && (
            <p className="text-[11px] text-slate-500 mb-2">
              → {node.children.length} child node{node.children.length > 1 ? 's' : ''} (see depth below)
            </p>
          )}

          {/* Accumulated iteration outputs */}
          {hasIterations && <IterationList iterations={node.iterations} />}

          {/* Final response for child nodes */}
          {!hasIterations && node.responsePreview && (
            <FinalOutputBlock text={node.responsePreview} />
          )}

          {/* Error */}
          {node.error && (
            <p className="mt-2 text-[12px] text-rose-300 bg-rose-950/30 px-3 py-2 rounded-lg border border-rose-800/40">
              ✗ {node.error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Helper: group nodes by depth ────────────────────────────── */

function groupByDepth(
  nodes: Record<string, TrajectoryNode>,
): { depth: number; nodeIds: string[] }[] {
  const depthMap = new Map<number, string[]>();

  for (const node of Object.values(nodes)) {
    const existing = depthMap.get(node.depth) ?? [];
    existing.push(node.id);
    depthMap.set(node.depth, existing);
  }

  return Array.from(depthMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([depth, nodeIds]) => ({
      depth,
      nodeIds: nodeIds.sort((a, b) => a.localeCompare(b)),
    }));
}

/* ── Vertical connector arrows between depth rows ────────────── */

function DepthConnector({ depth }: { depth: number }) {
  return (
    <div className="flex items-center justify-center py-1">
      <div className="flex flex-col items-center">
        <div className="w-px h-3 bg-slate-700/40" />
        <div className="text-slate-700 text-[10px]">↓</div>
      </div>
    </div>
  );
}

/* ── Main panel ────────────────────────────────────────────────── */

export function TrajectoryGraphPanel({ nodes, collapsedNodeIds, onToggleNode }: TrajectoryGraphPanelProps) {
  const depthGroups = groupByDepth(nodes);
  const scrollRef = useRef<HTMLDivElement>(null);
  const nodeCountRef = useRef(0);

  // Auto-scroll to bottom when new depth levels appear
  useEffect(() => {
    const count = Object.keys(nodes).length;
    if (count > nodeCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
    nodeCountRef.current = count;
  }, [nodes]);

  return (
    <section className="flex flex-col h-full bg-[#0a0e1a] overflow-hidden">
      <div className="shrink-0 border-b border-slate-800/60 px-5 py-3 flex items-center justify-between">
        <h2 className="text-[13px] font-semibold text-slate-200 tracking-tight">
          Trajectory Graph
        </h2>
        <span className="text-[11px] font-mono text-slate-500">
          {Object.keys(nodes).length} node{Object.keys(nodes).length !== 1 ? 's' : ''}
          {depthGroups.length > 0 && ` · ${depthGroups.length} depth${depthGroups.length > 1 ? 's' : ''}`}
        </span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto p-5">
        {depthGroups.length === 0 ? (
          <div className="grid h-full min-h-[200px] place-items-center rounded-xl border border-dashed border-slate-700/40 bg-slate-900/20">
            <div className="text-center space-y-3">
              <div className="text-3xl text-slate-700">⟡</div>
              <p className="text-[13px] text-slate-500">Waiting for node stream…</p>
              <p className="text-[12px] text-slate-600">Run an RLM to see the trajectory tree</p>
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            {depthGroups.map(({ depth, nodeIds }, groupIdx) => (
              <div key={depth}>
                {/* Connector arrow between depths */}
                {groupIdx > 0 && <DepthConnector depth={depth} />}

                {/* Depth level header */}
                <div className="flex items-center gap-2 mb-2.5 px-1">
                  <span
                    className={`text-[11px] font-semibold uppercase tracking-wider ${depthColor(depth)}`}
                  >
                    Depth {depth}
                  </span>
                  <div className="flex-1 h-px bg-slate-800/40" />
                  <span className="text-[11px] text-slate-600 font-mono">
                    {nodeIds.length} node{nodeIds.length > 1 ? 's' : ''}
                  </span>
                </div>

                {/* Horizontal scroll row of node cards */}
                <div
                  className={`flex gap-3 overflow-x-auto pb-2 rounded-xl border p-3 ${depthBg(depth)} ${depthAccent(depth)}`}
                  style={{ scrollbarWidth: 'thin' }}
                >
                  {nodeIds.map((nodeId) => {
                    const node = nodes[nodeId];
                    if (!node) return null;
                    const isExpanded = !collapsedNodeIds.has(nodeId);
                    return (
                      <NodeCard
                        key={nodeId}
                        node={node}
                        nodes={nodes}
                        isExpanded={isExpanded}
                        onToggle={() => onToggleNode(nodeId)}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
