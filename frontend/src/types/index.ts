// API response types

export interface DataInfo {
  n_times: number;
  n_wavelengths: number;
  time_range: [number, number];
  wavelength_range: [number, number];
  n_times_work: number;
  n_wavelengths_work: number;
  time_work_range: [number, number];
  wavelength_work_range: [number, number];
}

export interface RawDataPlot {
  wavelengths: number[];
  times: number[];
  data: number[][];
  z_min: number;
  z_max: number;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  info: DataInfo;
  raw_data: RawDataPlot;
}

export interface SVDResponse {
  singular_values: number[];
  wLSV: number[][];
  times: number[];
  n_components: number;
}

export interface GAResponse {
  taus: number[];
  DAS: number[][];
  fit: number[][];
  wLSV_fit: number[][];
  wLSV_indices: number[];
  times: number[];
  fwhm_fitted: number | null;
}

export interface LCurveData {
  x: number[];
  y: number[];
  curvature: number[];
  optimal_idx: number;
}

export interface LDMData {
  data: number[][][];
  ga_taus: number[] | null;
  z_min: number;
  z_max: number;
}

export interface LDAResponse {
  gcvs?: number[];
  cps: number[];
  lcurve: LCurveData;
  ldm: LDMData;
  alphas: number[];
  taus: number[];
  wavelengths: number[];
}

export interface TSVDResponse {
  ldm: number[][];
  taus: number[];
  wavelengths: number[];
  k: number;
  ga_taus: number[] | null;
  z_min: number;
  z_max: number;
}

export interface ChirpFitResponse {
  wavelengths: number[];
  delay_shift: number[];
  fitted_chirp: number[];
  irf: {
    munot: number;
    lamnot: number;
    mus: number[];
  };
}

// Component props types

export interface SessionState {
  sessionId: string | null;
  filename: string | null;
  dataInfo: DataInfo | null;
  rawData: RawDataPlot | null;
}
