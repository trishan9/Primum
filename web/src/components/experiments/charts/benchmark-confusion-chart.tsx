"use client";

import { useMemo } from "react";
import { HeatMap, type CustomLayerProps } from "@nivo/heatmap";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { inkOn } from "@/components/charts/colors";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme } from "@/components/charts/theme";
import { DECISIONS, type Confusion } from "@/lib/charts/pubmedqa";
import { fmtInt, fmtPct } from "@/lib/format";

import { CONFUSION_HEIGHT } from "./benchmark-heights";

interface Cell {
  x: string;
  /** the colour: this cell's share of its gold row */
  y: number | null;
  count: number;
}
type Row = { id: string; data: Cell[] };

/** d3 colour scales return `rgb(r, g, b)`; `inkOn` reads hex, so convert first. */
function toHex(color: string): string {
  const m = color.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!m) return color;
  return `#${[m[1], m[2], m[3]].map((c) => Number(c).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * Gold label (rows) against the generator's decision (columns). Colour is the cell's share of
 * its gold row on one sequential hue, so rows of different sizes read alike; each cell prints its
 * count and that share. The diagonal is agreement.
 */
export function ConfusionChart({ confusion, ariaLabel }: { confusion: Confusion; ariaLabel: string }) {
  const kit = useChartTheme();
  const data = useMemo<Row[]>(
    () =>
      DECISIONS.map((gold) => ({
        id: gold,
        data: DECISIONS.map((decision) => {
          const cell = confusion.cells.find((c) => c.gold === gold && c.decision === decision);
          return { x: decision, y: cell?.shareOfGold ?? null, count: cell?.count ?? 0 };
        }),
      })),
    [confusion],
  );
  if (!kit) return <ChartSkeleton height={CONFUSION_HEIGHT} />;
  const { theme, tokens, animate, colors } = kit;
  const fill = colors.sequential([0, 1]);

  // Two-line cell labels: the count, then the row share.
  function Labels({ cells }: CustomLayerProps<Cell>) {
    return (
      <g aria-hidden="true">
        {cells.map((cell) => {
          const ink = inkOn(tokens, toHex(cell.color));
          return (
            <g key={cell.id} transform={`translate(${cell.x}, ${cell.y})`} style={{ pointerEvents: "none" }}>
              <text textAnchor="middle" dy="-0.15em" fill={ink} fontSize={14} fontWeight={600} style={{ fontVariantNumeric: "tabular-nums" }}>
                {fmtInt(cell.data.count)}
              </text>
              <text textAnchor="middle" dy="1.15em" fill={ink} fontSize={12} style={{ fontVariantNumeric: "tabular-nums" }}>
                {cell.value === null ? "–" : fmtPct(cell.value, 0)}
              </text>
            </g>
          );
        })}
      </g>
    );
  }

  return (
    <Measured height={CONFUSION_HEIGHT}>
      {(width) => (
        <HeatMap<Cell, object>
          width={width}
          height={CONFUSION_HEIGHT}
          data={data}
          margin={{ top: 44, right: 8, bottom: 8, left: 76 }}
          valueFormat={(v) => fmtPct(v, 0)}
          colors={(cell) => (cell.value === null ? tokens.raised : fill(cell.value))}
          emptyColor={tokens.raised}
          xInnerPadding={0.04}
          yInnerPadding={0.06}
          borderRadius={4}
          enableLabels={false}
          axisTop={{ tickSize: 0, tickPadding: 8, legend: "Generator's decision", legendPosition: "middle", legendOffset: -34 }}
          axisLeft={{ tickSize: 0, tickPadding: 10, legend: "Gold label", legendPosition: "middle", legendOffset: -64 }}
          axisRight={null}
          axisBottom={null}
          hoverTarget="cell"
          inactiveOpacity={1}
          activeOpacity={1}
          layers={["grid", "axes", "cells", Labels]}
          theme={theme}
          animate={animate}
          role="img"
          ariaLabel={ariaLabel}
          tooltip={({ cell }) => (
            <div style={theme.tooltip?.container}>
              <p style={{ color: tokens.muted }}>
                Gold {cell.serieId}, answered {cell.data.x}
              </p>
              <p>
                <strong>{fmtInt(cell.data.count)}</strong> questions, {cell.value === null ? "–" : fmtPct(cell.value, 1)} of gold {cell.serieId}
              </p>
            </div>
          )}
        />
      )}
    </Measured>
  );
}
