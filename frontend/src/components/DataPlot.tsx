import Plot from 'react-plotly.js';
import type { RawDataPlot } from '../types';

interface DataPlotProps {
  data: RawDataPlot;
  title?: string;
}

export function DataPlot({ data, title = 'Raw Data' }: DataPlotProps) {
  return (
    <Plot
      data={[
        {
          type: 'contour',
          x: data.wavelengths,
          y: data.times,
          z: data.data,
          colorscale: 'RdBu',
          reversescale: true,
          zmin: data.z_min,
          zmax: data.z_max,
          colorbar: {
            title: { text: 'Intensity', side: 'right' },
          },
        },
      ]}
      layout={{
        title: { text: title, font: { size: 16 } },
        xaxis: { title: { text: 'Wavelength (nm)' } },
        yaxis: { title: { text: 'Time' }, type: 'log' },
        autosize: true,
        margin: { l: 60, r: 50, t: 50, b: 50 },
      }}
      style={{ width: '100%', height: '500px' }}
      config={{ responsive: true }}
      useResizeHandler={true}
    />
  );
}
