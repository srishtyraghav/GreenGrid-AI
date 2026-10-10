import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const fetchMetadata = async () => {
  const response = await axios.get(`${API_BASE_URL}/metadata`);
  return response.data;
};

export const fetchPriorityZones = async (year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase7/zones/${year}/${scenario}`);
  return response.data;
};

export const fetchTreeRequirementSummary = async (year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase8/summary/${year}/${scenario}`);
  return response.data;
};

export const fetchTreeRequirementZones = async (year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase8/zones/${year}/${scenario}`);
  return response.data;
};

export const fetchTreeVectors = async (layer: string, year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase8/vectors/${layer}/${year}/${scenario}`);
  return response.data;
};

export const fetchBoundary = async () => {
  const response = await axios.get(`${API_BASE_URL}/boundary`);
  return response.data;
};

export const fetchOsmReferenceBoundary = async () => {
  const response = await axios.get(`${API_BASE_URL}/boundary/osm_reference`);
  return response.data;
};

export const fetchClassShares = async (year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase7/class_shares/${year}/${scenario}`);
  return response.data;
};

export const fetchExclusions = async (year: number, scenario: string) => {
  const response = await axios.get(`${API_BASE_URL}/phase7/exclusions/${year}/${scenario}`);
  return response.data;
};

export const getTileUrl = (phase: string, layer: string, year: number) => {
  return `${API_BASE_URL}/tiles/${phase}/${layer}/${year}/{z}/{x}/{y}.png`;
};
