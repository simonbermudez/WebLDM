import { useState } from 'react';
import Plot from 'react-plotly.js';
import type { GAResponse } from '../types';

interface GAPanelProps {
  sessionId: string;
  onRunGA: (
    wLSVs: string,
    initialTaus: number[],
    bounds: [number, number][],
    alpha: number,
    fitIRF: boolean
  ) => Promise<void>;
  gaResults: GAResponse | null;
  isLoading: boolean;
}

export function GAPanel({ onRunGA, gaResults, isLoading }: GAPanelProps) {
  const [numTaus, setNumTaus] = useState(3);
  const [wLSVs, setWLSVs] = useState('3');
  const [initialTaus, setInitialTaus] = useState<string[]>(['1', '10', '100']);
  const [bounds, setBounds] = useState<[string, string][]>([
    ['0.1', '10'],
    ['1', '100'],
    ['10', '1000'],
  ]);
  const [alpha, setAlpha] = useState(0);
  const [fitIRF, setFitIRF] = useState(false);
  const [selectedFitIdx, setSelectedFitIdx] = useState(0);

  const handleNumTausChange = (n: number) => {
    setNumTaus(n);
    const newTaus = Array.from({ length: n }, (_, i) =>
      Math.pow(10, i).toString()
    );
    const newBounds: [string, string][] = Array.from({ length: n }, (_, i) => [
      Math.pow(10, i - 1).toString(),
      Math.pow(10, i + 1).toString(),
    ]);
    setInitialTaus(newTaus);
    setBounds(newBounds);
  };

  const handleRun = async () => {
    const taus = initialTaus.map((t) => parseFloat(t));
    const b = bounds.map(
      ([min, max]) => [parseFloat(min), parseFloat(max)] as [number, number]
    );
    await onRunGA(wLSVs, taus, b, alpha, fitIRF);
  };

  return (
    <div className="ga-panel">
      <h3>Global Analysis</h3>

      <div className="ga-controls">
        <div className="control-group">
          <label>Number of Lifetimes:</label>
          <input
            type="number"
            min={1}
            max={10}
            value={numTaus}
            onChange={(e) => handleNumTausChange(parseInt(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>wLSVs to fit (space-separated or single number):</label>
          <input
            type="text"
            value={wLSVs}
            onChange={(e) => setWLSVs(e.target.value)}
            placeholder="e.g., 3 or 1 2 3"
          />
        </div>

        <div className="tau-inputs">
          <h4>Initial Lifetimes & Bounds</h4>
          {initialTaus.map((tau, i) => (
            <div key={i} className="tau-row">
              <span>tau_{i + 1}:</span>
              <input
                type="number"
                value={tau}
                onChange={(e) => {
                  const newTaus = [...initialTaus];
                  newTaus[i] = e.target.value;
                  setInitialTaus(newTaus);
                }}
                placeholder="Initial"
              />
              <input
                type="number"
                value={bounds[i][0]}
                onChange={(e) => {
                  const newBounds = [...bounds];
                  newBounds[i] = [e.target.value, bounds[i][1]];
                  setBounds(newBounds);
                }}
                placeholder="Min"
              />
              <input
                type="number"
                value={bounds[i][1]}
                onChange={(e) => {
                  const newBounds = [...bounds];
                  newBounds[i] = [bounds[i][0], e.target.value];
                  setBounds(newBounds);
                }}
                placeholder="Max"
              />
            </div>
          ))}
        </div>

        <div className="control-group">
          <label>Alpha (regularization):</label>
          <input
            type="number"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            step={0.1}
          />
        </div>

        <div className="control-group checkbox">
          <label>
            <input
              type="checkbox"
              checked={fitIRF}
              onChange={(e) => setFitIRF(e.target.checked)}
            />
            Fit IRF (FWHM)
          </label>
        </div>

        <button onClick={handleRun} disabled={isLoading}>
          {isLoading ? 'Running...' : 'Run Global Analysis'}
        </button>
      </div>

      {gaResults && (
        <div className="ga-results">
          <div className="fitted-taus">
            <h4>Fitted Lifetimes:</h4>
            <ul>
              {gaResults.taus.map((tau, i) => (
                <li key={i}>
                  tau_{i + 1} = {tau.toFixed(3)}
                </li>
              ))}
            </ul>
            {gaResults.fwhm_fitted && (
              <p>Fitted FWHM: {gaResults.fwhm_fitted.toFixed(3)}</p>
            )}
          </div>

          <div className="plot-row">
            <Plot
              data={gaResults.DAS.map((das, i) => ({
                type: 'scatter' as const,
                mode: 'lines' as const,
                x: gaResults.wLSV_indices,
                y: das,
                name: `tau = ${gaResults.taus[i].toFixed(2)}`,
              }))}
              layout={{
                title: { text: 'Decay Associated Spectra' },
                xaxis: { title: { text: 'wLSV Index' } },
                yaxis: { title: { text: 'Amplitude' } },
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 50 },
              }}
              style={{ width: '100%', maxWidth: '400px', height: '350px' }}
              config={{ responsive: true }}
              useResizeHandler={true}
            />

            <div className="fit-plot">
              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'markers',
                    x: gaResults.times,
                    y: gaResults.wLSV_fit.map((row) => row[selectedFitIdx]),
                    name: 'Data',
                    marker: { color: '#1f77b4', size: 6 },
                  },
                  {
                    type: 'scatter',
                    mode: 'lines',
                    x: gaResults.times,
                    y: gaResults.fit.map((row) => row[selectedFitIdx]),
                    name: 'Fit',
                    line: { color: '#d62728', width: 2 },
                  },
                ]}
                layout={{
                  title: { text: `wLSV ${gaResults.wLSV_indices[selectedFitIdx]} Fit` },
                  xaxis: { title: { text: 'Time' }, type: 'log' },
                  yaxis: { title: { text: 'Amplitude' } },
                  autosize: true,
                  margin: { l: 60, r: 30, t: 50, b: 50 },
                }}
                style={{ width: '100%', maxWidth: '500px', height: '350px' }}
                config={{ responsive: true }}
                useResizeHandler={true}
              />
              <div className="slider-container">
                <label>wLSV: {gaResults.wLSV_indices[selectedFitIdx]}</label>
                <input
                  type="range"
                  min={0}
                  max={gaResults.wLSV_indices.length - 1}
                  value={selectedFitIdx}
                  onChange={(e) => setSelectedFitIdx(parseInt(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
