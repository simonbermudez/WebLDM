import { useState } from 'react';
import Plot from 'react-plotly.js';
import type { SVDResponse } from '../types';

interface SVDPanelProps {
  data: SVDResponse;
}

export function SVDPanel({ data }: SVDPanelProps) {
  const [selectedLSV, setSelectedLSV] = useState(0);

  return (
    <div className="svd-panel">
      <div className="plot-row">
        <Plot
          data={[
            {
              type: 'scatter',
              mode: 'lines+markers',
              x: Array.from({ length: data.singular_values.length }, (_, i) => i + 1),
              y: data.singular_values,
              marker: { color: '#1f77b4' },
            },
          ]}
          layout={{
            title: { text: 'Singular Values' },
            xaxis: { title: { text: 'Component' } },
            yaxis: { title: { text: 'Singular Value' }, type: 'log' },
            width: 400,
            height: 350,
            margin: { l: 60, r: 30, t: 50, b: 50 },
          }}
        />

        <div className="wlsv-plot">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: data.times,
                y: data.wLSV.map((row) => row[selectedLSV]),
                line: { color: '#1f77b4' },
                name: `wLSV ${selectedLSV + 1}`,
              },
            ]}
            layout={{
              title: { text: `Weighted Left Singular Vector ${selectedLSV + 1}` },
              xaxis: { title: { text: 'Time' }, type: 'log' },
              yaxis: { title: { text: 'Amplitude' } },
              width: 500,
              height: 350,
              margin: { l: 60, r: 30, t: 50, b: 50 },
            }}
          />
          <div className="slider-container">
            <label>wLSV: {selectedLSV + 1}</label>
            <input
              type="range"
              min={0}
              max={Math.min(data.n_components - 1, 9)}
              value={selectedLSV}
              onChange={(e) => setSelectedLSV(parseInt(e.target.value))}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
