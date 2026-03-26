import { useEffect, useState, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { Search, SlidersHorizontal, X, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import ProductCard from "@/components/ProductCard";
import { api } from "@/lib/api";
import type { Product, FilterOptions } from "@/types";
import { formatPrice } from "@/lib/utils";

const SORT_OPTIONS = [
  { value: "created_desc", label: "Mới nhất" },
  { value: "rating_desc", label: "Đánh giá cao nhất" },
  { value: "price_asc", label: "Giá tăng dần" },
  { value: "price_desc", label: "Giá giảm dần" },
  { value: "discount_desc", label: "Khuyến mãi nhiều nhất" },
  { value: "name_asc", label: "Tên A-Z" },
];

export default function ProductListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [products, setProducts] = useState<Product[]>([]);
  const [filterOptions, setFilterOptions] = useState<FilterOptions | null>(null);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [filtersVisible, setFiltersVisible] = useState(true);

  // Derived state from URL params
  const search = searchParams.get("search") || "";
  const page = Number(searchParams.get("page") || "1");
  const sortValue = searchParams.get("sort") || "created_desc";
  const selectedBrands = searchParams.getAll("brands");
  const selectedCategories = searchParams.getAll("categories");
  const minPriceParam = searchParams.get("min_price");
  const maxPriceParam = searchParams.get("max_price");

  // Local search input
  const [searchInput, setSearchInput] = useState(search);

  // Price range local state (synced from filterOptions)
  const [priceRange, setPriceRange] = useState<[number, number]>([0, 10000000]);

  function updateParams(updates: Record<string, string | string[] | null>) {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      for (const [key, value] of Object.entries(updates)) {
        if (value === null || (Array.isArray(value) && value.length === 0)) {
          next.delete(key);
        } else if (Array.isArray(value)) {
          next.delete(key);
          value.forEach((v) => next.append(key, v));
        } else {
          next.set(key, value);
        }
      }
      // Reset page on filter change
      if (!("page" in updates)) next.set("page", "1");
      return next;
    });
  }

  // Load filter options once
  useEffect(() => {
    api.getFilterOptions().then((opts) => {
      setFilterOptions(opts);
      const min = minPriceParam ? Number(minPriceParam) : opts.min_price;
      const max = maxPriceParam ? Number(maxPriceParam) : opts.max_price;
      setPriceRange([min, max]);
    }).catch(() => {});
  }, []);

  // Sync priceRange when URL params change
  useEffect(() => {
    if (filterOptions) {
      setPriceRange([
        minPriceParam ? Number(minPriceParam) : filterOptions.min_price,
        maxPriceParam ? Number(maxPriceParam) : filterOptions.max_price,
      ]);
    }
  }, [minPriceParam, maxPriceParam, filterOptions]);

  const fetchProducts = useCallback(async () => {
    setLoading(true);
    const [sort_by, sort_order] = sortValue.split("_") as [string, string];
    try {
      const res = await api.getProducts({
        page,
        page_size: 12,
        search: search || undefined,
        brands: selectedBrands.length ? selectedBrands : undefined,
        categories: selectedCategories.length ? selectedCategories : undefined,
        min_price: minPriceParam ? Number(minPriceParam) : undefined,
        max_price: maxPriceParam ? Number(maxPriceParam) : undefined,
        sort_by,
        sort_order,
      });
      setProducts(res.items);
      setTotal(res.total);
      setTotalPages(res.total_pages);
    } catch {
      setProducts([]);
    } finally {
      setLoading(false);
    }
  }, [searchParams.toString()]);

  useEffect(() => {
    fetchProducts();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [fetchProducts]);

  function toggleBrand(brand: string) {
    const next = selectedBrands.includes(brand)
      ? selectedBrands.filter((b) => b !== brand)
      : [...selectedBrands, brand];
    updateParams({ brands: next });
  }

  function toggleCategory(cat: string) {
    const next = selectedCategories.includes(cat)
      ? selectedCategories.filter((c) => c !== cat)
      : [...selectedCategories, cat];
    updateParams({ categories: next });
  }

  function applyPriceRange() {
    updateParams({
      min_price: String(priceRange[0]),
      max_price: String(priceRange[1]),
    });
  }

  function clearAllFilters() {
    setSearchParams({ page: "1" });
    if (filterOptions) setPriceRange([filterOptions.min_price, filterOptions.max_price]);
  }

  const hasActiveFilters = selectedBrands.length > 0 || selectedCategories.length > 0 || minPriceParam || maxPriceParam;

  const FilterPanel = () => (
    <div className="space-y-6">
      {/* Clear all */}
      {hasActiveFilters && (
        <Button variant="ghost" size="sm" onClick={clearAllFilters} className="text-red-500 hover:text-red-600 -ml-2">
          <X className="h-4 w-4 mr-1" /> Xóa tất cả bộ lọc
        </Button>
      )}

      {/* Brands */}
      {filterOptions && (
        <div>
          <h3 className="text-xs font-semibold text-[#0c1638] uppercase tracking-[0.06em] mb-3">Thương Hiệu</h3>
          <div className="space-y-2">
            {filterOptions.brands.map((brand) => (
              <label key={brand} className="flex items-center gap-2 cursor-pointer group">
                <Checkbox
                  checked={selectedBrands.includes(brand)}
                  onCheckedChange={() => toggleBrand(brand)}
                />
                <span className="text-sm text-[#444956] group-hover:text-[#0d1b67]">{brand}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      <Separator />

      {/* Categories */}
      {filterOptions && (
        <div>
          <h3 className="text-xs font-semibold text-[#0c1638] uppercase tracking-[0.06em] mb-3">Danh Mục</h3>
          <div className="space-y-2">
            {filterOptions.categories.map((cat) => (
              <label key={cat} className="flex items-center gap-2 cursor-pointer group">
                <Checkbox
                  checked={selectedCategories.includes(cat)}
                  onCheckedChange={() => toggleCategory(cat)}
                />
                <span className="text-sm text-[#444956] group-hover:text-[#0d1b67]">{cat}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      <Separator />

      {/* Price Range */}
      {filterOptions && (
        <div>
          <h3 className="text-xs font-semibold text-[#0c1638] uppercase tracking-[0.06em] mb-3">Khoảng Giá</h3>
          <Slider
            min={filterOptions.min_price}
            max={filterOptions.max_price}
            step={100000}
            value={priceRange}
            onValueChange={(v) => setPriceRange(v as [number, number])}
            className="mb-4"
          />
          <div className="flex justify-between text-xs text-[#aeaeae] mb-3">
            <span>{formatPrice(priceRange[0])}</span>
            <span>{formatPrice(priceRange[1])}</span>
          </div>
          <Button size="sm" variant="outline" onClick={applyPriceRange} className="w-full border-[#0d1b67]/20 text-[#0d1b67] hover:bg-[#0d1b67]/5">
            Áp dụng
          </Button>
        </div>
      )}
    </div>
  );

  return (
    <div className="bg-[#f7f7f7] min-h-screen">
      <div className="container mx-auto px-4 py-10">
        {/* Page Title */}
        <div className="mb-8">
          <h1 className="text-[56px] font-black tracking-[-0.04em] leading-none text-[#0d1b67]">
            {search ? "Tìm Kiếm" : "Tất Cả Sản Phẩm"}
          </h1>
          {search && (
            <p className="text-[#aeaeae] text-base mt-2">
              Kết quả cho: <span className="font-semibold text-[#444956]">"{search}"</span>
            </p>
          )}
        </div>

        {/* Search + controls bar */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              updateParams({ search: searchInput || null });
            }}
            className="flex-1 relative"
          >
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-[#aeaeae]" />
            <Input
              placeholder="Tìm kiếm sản phẩm..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="pl-10 h-10 rounded-full bg-[#ededed] border-0 text-[#444956] placeholder:text-[#aeaeae] focus-visible:ring-1 focus-visible:ring-[#0d1b67]/30"
            />
          </form>

          <div className="flex items-center gap-3">
            {/* Desktop: Hide/Show Filters toggle */}
            <button
              className="hidden md:flex items-center gap-2 text-[#0c1638] font-medium text-sm hover:opacity-70 transition-opacity"
              onClick={() => setFiltersVisible(!filtersVisible)}
            >
              <SlidersHorizontal className="h-4 w-4" />
              {filtersVisible ? "Ẩn bộ lọc" : "Hiện bộ lọc"}
            </button>

            {/* Mobile filter toggle */}
            <Button
              variant="ghost"
              className="md:hidden flex items-center gap-2 text-[#0c1638] font-medium"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <SlidersHorizontal className="h-4 w-4" />
              Lọc
              {hasActiveFilters && (
                <span className="bg-[#0d1b67] text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  {selectedBrands.length + selectedCategories.length + (minPriceParam ? 1 : 0)}
                </span>
              )}
            </Button>

            {/* Sort */}
            <Select
              value={sortValue}
              onValueChange={(v) => updateParams({ sort: v })}
            >
              <SelectTrigger className="w-[190px] bg-white rounded-full border-0 shadow-sm text-[#0c1638] font-medium text-sm">
                <SelectValue placeholder="Sắp xếp" />
              </SelectTrigger>
              <SelectContent>
                {SORT_OPTIONS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Active filter tags */}
        {hasActiveFilters && (
          <div className="flex flex-wrap gap-2 mb-5">
            {selectedBrands.map((b) => (
              <span key={b} className="inline-flex items-center gap-1 bg-[#0d1b67]/10 text-[#0d1b67] text-xs px-3 py-1 rounded-full border border-[#0d1b67]/20">
                {b}
                <button onClick={() => toggleBrand(b)}><X className="h-3 w-3" /></button>
              </span>
            ))}
            {selectedCategories.map((c) => (
              <span key={c} className="inline-flex items-center gap-1 bg-green-50 text-green-700 text-xs px-3 py-1 rounded-full border border-green-200">
                {c}
                <button onClick={() => toggleCategory(c)}><X className="h-3 w-3" /></button>
              </span>
            ))}
            {(minPriceParam || maxPriceParam) && (
              <span className="inline-flex items-center gap-1 bg-orange-50 text-orange-700 text-xs px-3 py-1 rounded-full border border-orange-200">
                Giá: {formatPrice(Number(minPriceParam || 0))} - {formatPrice(Number(maxPriceParam || filterOptions?.max_price || 0))}
                <button onClick={() => updateParams({ min_price: null, max_price: null })}><X className="h-3 w-3" /></button>
              </span>
            )}
          </div>
        )}

        <div className="flex gap-6">
          {/* Desktop Sidebar */}
          {filtersVisible && (
            <aside className="hidden md:block w-56 shrink-0">
              <div className="bg-white rounded-2xl p-5 sticky top-20 shadow-[0_4px_20px_rgba(0,0,0,0.06)]">
                <FilterPanel />
              </div>
            </aside>
          )}

          {/* Mobile Sidebar Overlay */}
          {sidebarOpen && (
            <div className="md:hidden fixed inset-0 z-50 flex">
              <div className="flex-1 bg-black/40" onClick={() => setSidebarOpen(false)} />
              <div className="w-72 bg-white h-full overflow-y-auto p-6">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="font-bold text-lg text-[#0d1b67]">Bộ Lọc</h2>
                  <button onClick={() => setSidebarOpen(false)}>
                    <X className="h-5 w-5 text-[#444956]" />
                  </button>
                </div>
                <FilterPanel />
              </div>
            </div>
          )}

          {/* Product Grid */}
          <div className="flex-1 min-w-0">
            {/* Results count */}
            <p className="text-sm text-[#aeaeae] mb-5">
              {loading ? "Đang tải..." : `${total} sản phẩm`}
            </p>

            {loading ? (
              <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i} className="bg-white rounded-xl aspect-[3/4] animate-pulse shadow-[0_4px_20px_rgba(0,0,0,0.06)]" />
                ))}
              </div>
            ) : products.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <div className="text-6xl mb-4">👟</div>
                <h3 className="text-lg font-semibold text-[#444956] mb-2">Không tìm thấy sản phẩm</h3>
                <p className="text-[#aeaeae] text-sm mb-6">Thử thay đổi bộ lọc hoặc từ khóa tìm kiếm</p>
                <Button variant="outline" onClick={clearAllFilters} className="border-[#0d1b67]/20 text-[#0d1b67] hover:bg-[#0d1b67]/5">Xóa bộ lọc</Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                {products.map((p) => <ProductCard key={p.id} product={p} />)}
              </div>
            )}

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-10">
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={page <= 1}
                  onClick={() => updateParams({ page: String(page - 1) })}
                  className="rounded-full text-[#0c1638] hover:bg-[#0d1b67]/10 disabled:opacity-30"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>

                {Array.from({ length: totalPages }, (_, i) => i + 1)
                  .filter((p) => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
                  .reduce<(number | "...")[]>((acc, p, i, arr) => {
                    if (i > 0 && p - (arr[i - 1] as number) > 1) acc.push("...");
                    acc.push(p);
                    return acc;
                  }, [])
                  .map((item, i) =>
                    item === "..." ? (
                      <span key={`ellipsis-${i}`} className="px-2 text-[#aeaeae]">...</span>
                    ) : (
                      <Button
                        key={item}
                        variant="ghost"
                        size="icon"
                        onClick={() => updateParams({ page: String(item) })}
                        className={`rounded-full text-sm font-medium transition-colors ${
                          item === page
                            ? "bg-[#0d1b67] text-white hover:bg-[#0d1b67]/90"
                            : "text-[#0c1638] hover:bg-[#0d1b67]/10"
                        }`}
                      >
                        {item}
                      </Button>
                    )
                  )}

                <Button
                  variant="ghost"
                  size="icon"
                  disabled={page >= totalPages}
                  onClick={() => updateParams({ page: String(page + 1) })}
                  className="rounded-full text-[#0c1638] hover:bg-[#0d1b67]/10 disabled:opacity-30"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
