import { useState } from 'react';
import Plot from 'react-plotly.js';
import type { LDAResponse } from '../types';

interface LDAPanelProps {
  sessionId: string;
  onRunLDA: (params: {
    tauMin: number;
    tauMax: number;
    nTaus: number;
    alphaMin: number;
    alphaMax: number;
    nAlphas: number;
    regMethod: string;
    regMatrix: string;
    simfit: boolean;
    gaTaus?: number[];
  }) => Promise<void>;
  ldaResults: LDAResponse | null;
  gaTaus?: number[];
  isLoading: boolean;
}

export function LDAPanel({
  onRunLDA,
  ldaResults,
  gaTaus,
  isLoading,
}: LDAPanelProps) {
  const [tauMin, setTauMin] = useState(-1);
  const [tauMax, setTauMax] = useState(4);
  const [nTaus, setNTaus] = useState(100);
  const [alphaMin, setAlphaMin] = useState(-4);
  const [alphaMax, setAlphaMax] = useState(2);
  const [nAlphas, setNAlphas] = useState(20);
  const [regMethod, setRegMethod] = useState('L2');
  const [regMatrix, setRegMatrix] = useState('Id');
  const [simfit] = useState(true);

  const [selectedAlpha, setSelectedAlpha] = useState(0);
  const [selectedWl, setSelectedWl] = useState(0);

  const handleRun = async () => {
    await onRunLDA({
      tauMin,
      tauMax,
      nTaus,
      alphaMin,
      alphaMax,
      nAlphas,
      regMethod,
      regMatrix,
      simfit,
      gaTaus,
    });
  };

  // Get LDM data for selected alpha
  const getLDMSlice = (): number[][] => {
    if (!ldaResults) return [];
    return ldaResults.ldm.data.map((row) => row.map((col) => col[selectedAlpha]));
  };

  // Get wavelength trace
  const getWlTrace = (): number[] => {
    if (!ldaResults) return [];
    return ldaResults.ldm.data.map((row) => row[selectedWl][selectedAlpha]);
  };

  return (
    <div className="lda-panel">
      <h3>Lifetime Density Analysis</h3>

      <div className="lda-controls">
        <div className="control-row">
          <div className="control-group">
            <label>Tau Range (log10):</label>
            <div className="range-inputs">
              <input
                type="number"
                value={tauMin}
                onChange={(e) => setTauMin(parseFloat(e.target.value))}
                step={0.5}
              />
              <span>to</span>
              <input
                type="number"
                value={tauMax}
                onChange={(e) => setTauMax(parseFloat(e.target.value))}
                step={0.5}
              />
              <span>({nTaus} points)</span>
              <input
                type="number"
                value={nTaus}
                onChange={(e) => setNTaus(parseInt(e.target.value))}
                min={10}
                max={500}
              />
            </div>
          </div>
        </div>

        <div className="control-row">
          <div className="control-group">
            <label>Alpha Range (log10):</label>
            <div className="range-inputs">
              <input
                type="number"
                value={alphaMin}
                onChange={(e) => setAlphaMin(parseFloat(e.target.value))}
                step={0.5}
              />
              <span>to</span>
              <input
                type="number"
                value={alphaMax}
                onChange={(e) => setAlphaMax(parseFloat(e.target.value))}
                step={0.5}
              />
              <span>({nAlphas} points)</span>
              <input
                type="number"
                value={nAlphas}
                onChange={(e) => setNAlphas(parseInt(e.target.value))}
                min={5}
                max={50}
              />
            </div>
          </div>
        </div>

        <div className="control-row">
          <div className="control-group">
            <label>Regularization Method:</label>
            <select value={regMethod} onChange={(e) => setRegMethod(e.target.value)}>
              <option value="L2">Tikhonov (L2)</option>
              <option value="L1">LASSO (L1)</option>
              <option value="elnet">Elastic Net</option>
            </select>
          </div>

          <div className="control-group">
            <label>Regularization Matrix:</label>
            <select value={regMatrix} onChange={(e) => setRegMatrix(e.target.value)}>
              <option value="Id">Identity</option>
              <option value="1D">1st Derivative</option>
              <option value="2D">2nd Derivative</option>
              <option value="Fused">Fused</option>
            </select>
          </div>
        </div>

        <button onClick={handleRun} disabled={isLoading}>
          {isLoading ? 'Running LDA...' : 'Run LDA'}
        </button>
      </div>

      {ldaResults && (
        <div className="lda-results">
          {/* L-curve and statistics */}
          <div className="plot-row">
            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: ldaResults.lcurve.x,
                  y: ldaResults.lcurve.y,
                  marker: { color: '#1f77b4' },
                },
                {
                  type: 'scatter',
                  mode: 'markers',
                  x: [ldaResults.lcurve.x[ldaResults.lcurve.optimal_idx]],
                  y: [ldaResults.lcurve.y[ldaResults.lcurve.optimal_idx]],
                  marker: { color: 'red', size: 12 },
                  name: 'Optimal',
                },
              ]}
              layout={{
                title: { text: 'L-Curve' },
                xaxis: { title: { text: 'Residual Norm' } },
                yaxis: { title: { text: 'Smoothing Norm' } },
                width: 350,
                height: 300,
                margin: { l: 60, r: 30, t: 50, b: 50 },
                showlegend: false,
              }}
            />

            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: ldaResults.alphas,
                  y: ldaResults.lcurve.curvature,
                  marker: { color: '#1f77b4' },
                },
                {
                  type: 'scatter',
                  mode: 'markers',
                  x: [ldaResults.alphas[ldaResults.lcurve.optimal_idx]],
                  y: [ldaResults.lcurve.curvature[ldaResults.lcurve.optimal_idx]],
                  marker: { color: 'red', size: 12 },
                },
              ]}
              layout={{
                title: { text: 'Curvature' },
                xaxis: { title: { text: 'Alpha' }, type: 'log' },
                yaxis: { title: { text: 'Curvature' } },
                width: 350,
                height: 300,
                margin: { l: 60, r: 30, t: 50, b: 50 },
                showlegend: false,
              }}
            />

            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: ldaResults.alphas,
                  y: ldaResults.cps,
                  marker: { color: '#1f77b4' },
                  name: 'Cp',
                },
                ...(ldaResults.gcvs
                  ? [
                      {
                        type: 'scatter' as const,
                        mode: 'lines+markers' as const,
                        x: ldaResults.alphas,
                        y: ldaResults.gcvs,
                        marker: { color: '#ff7f0e' },
                        name: 'GCV',
                        yaxis: 'y2' as const,
                      },
                    ]
                  : []),
              ]}
              layout={{
                title: { text: 'Cp / GCV' },
                xaxis: { title: { text: 'Alpha' }, type: 'log' },
                yaxis: { title: { text: 'Cp' } },
                yaxis2: ldaResults.gcvs
                  ? {
                      title: { text: 'GCV' },
                      overlaying: 'y' as const,
                      side: 'right' as const,
                    }
                  : undefined,
                width: 350,
                height: 300,
                margin: { l: 60, r: 60, t: 50, b: 50 },
              }}
            />
          </div>

          {/* LDM Plot */}
          <div className="ldm-section">
            <h4>Lifetime Density Map (Alpha = {ldaResults.alphas[selectedAlpha].toExponential(2)})</h4>
            <div className="plot-row">
              <Plot
                data={[
                  {
                    type: 'contour',
                    x: ldaResults.wavelengths,
                    y: ldaResults.taus,
                    z: getLDMSlice(),
                    colorscale: 'RdBu',
                    reversescale: true,
                    colorbar: {
                      title: { text: 'Amplitude' },
                    },
                  },
                  // Add GA tau lines if available
                  ...(gaTaus
                    ? gaTaus.map((tau) => ({
                        type: 'scatter' as const,
                        mode: 'lines' as const,
                        x: [ldaResults.wavelengths[0], ldaResults.wavelengths[ldaResults.wavelengths.length - 1]],
                        y: [tau, tau],
                        line: { color: 'black', dash: 'dash' as const, width: 2 },
                        showlegend: false,
                      }))
                    : []),
                ]}
                layout={{
                  xaxis: { title: { text: 'Wavelength (nm)' } },
                  yaxis: { title: { text: 'Lifetime (tau)' }, type: 'log' },
                  width: 600,
                  height: 450,
                  margin: { l: 60, r: 80, t: 30, b: 50 },
                }}
              />

              <Plot
                data={[
                  {
                    type: 'scatter',
                    mode: 'lines',
                    x: ldaResults.taus,
                    y: getWlTrace(),
                    line: { color: '#1f77b4' },
                  },
                ]}
                layout={{
                  title: { text: `Wavelength = ${ldaResults.wavelengths[selectedWl].toFixed(1)} nm` },
                  xaxis: { title: { text: 'Lifetime (tau)' }, type: 'log' },
                  yaxis: { title: { text: 'Amplitude' } },
                  width: 400,
                  height: 450,
                  margin: { l: 60, r: 30, t: 50, b: 50 },
                }}
              />
            </div>

            <div className="ldm-sliders">
              <div className="slider-container">
                <label>Alpha: {ldaResults.alphas[selectedAlpha].toExponential(2)}</label>
                <input
                  type="range"
                  min={0}
                  max={ldaResults.alphas.length - 1}
                  value={selectedAlpha}
                  onChange={(e) => setSelectedAlpha(parseInt(e.target.value))}
                />
              </div>
              <div className="slider-container">
                <label>Wavelength: {ldaResults.wavelengths[selectedWl].toFixed(1)} nm</label>
                <input
                  type="range"
                  min={0}
                  max={ldaResults.wavelengths.length - 1}
                  value={selectedWl}
                  onChange={(e) => setSelectedWl(parseInt(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
