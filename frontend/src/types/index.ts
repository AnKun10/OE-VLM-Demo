export interface Product {
  id: string;
  name: string;
  image_url: string;
  store: string;
  layer: string;
  category: string;
  description: string;
}

export interface ProductListResponse {
  items: Product[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface FilterOptions {
  stores: string[];
  categories: string[];
  layers: string[];
}
