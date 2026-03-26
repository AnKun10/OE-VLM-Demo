export interface ProductColor {
  name: string;
  hex: string;
  image_url?: string;
}

export interface Product {
  id: string;
  name: string;
  brand: string;
  category: string;
  description: string;
  price: number;
  original_price?: number;
  discount_percent?: number;
  colors: ProductColor[];
  sizes: string[];
  images: string[];
  tags: string[];
  is_new: boolean;
  in_stock: boolean;
  stock_qty: number;
  rating: number;
  review_count: number;
}

export interface ProductListResponse {
  items: Product[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface FilterOptions {
  brands: string[];
  categories: string[];
  min_price: number;
  max_price: number;
}
