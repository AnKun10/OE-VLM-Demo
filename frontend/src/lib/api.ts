import type { Product, ProductListResponse, FilterOptions } from "@/types";

const BASE_URL = "/api";

export interface ProductsParams {
  page?: number;
  page_size?: number;
  search?: string;
  stores?: string[];
  layers?: string[];
  categories?: string[];
  sort_by?: string;
  sort_order?: string;
  semantic?: boolean;
}

function buildQuery(params: Record<string, unknown>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      value.forEach((v) => qs.append(key, String(v)));
    } else {
      qs.set(key, String(value));
    }
  }
  return qs.toString();
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  getProducts(params: ProductsParams = {}): Promise<ProductListResponse> {
    const qs = buildQuery(params as Record<string, unknown>);
    return fetchJSON(`${BASE_URL}/products${qs ? `?${qs}` : ""}`);
  },

  getProduct(id: string): Promise<Product> {
    return fetchJSON(`${BASE_URL}/products/${id}`);
  },

  getRelatedProducts(id: string, limit = 4): Promise<Product[]> {
    return fetchJSON(`${BASE_URL}/products/${id}/related?limit=${limit}`);
  },

  getFilterOptions(): Promise<FilterOptions> {
    return fetchJSON(`${BASE_URL}/products/filters`);
  },
};
