import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Search, Zap, Shield, Truck, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ProductCard from "@/components/ProductCard";
import { api } from "@/lib/api";
import type { Product } from "@/types";

const STORES = [
  { name: "AnStore", color: "bg-orange-100 text-orange-700 hover:bg-orange-200" },
  { name: "ThanhStore", color: "bg-blue-100 text-blue-700 hover:bg-blue-200" },
  { name: "TuanAnhStore", color: "bg-teal-100 text-teal-700 hover:bg-teal-200" },
];

const FEATURES = [
  { icon: Shield, title: "Hàng Chính Hãng 100%", desc: "Cam kết sản phẩm chính hãng từ nhà phân phối uy tín" },
  { icon: Truck, title: "Giao Hàng Toàn Quốc", desc: "Giao hàng nhanh trong 2-3 ngày làm việc" },
  { icon: RotateCcw, title: "Đổi Trả 30 Ngày", desc: "Đổi trả dễ dàng trong vòng 30 ngày nếu không hài lòng" },
  { icon: Zap, title: "Tư Vấn Chuyên Nghiệp", desc: "Đội ngũ chuyên gia giày chạy bộ sẵn sàng hỗ trợ bạn" },
];

export default function HomePage() {
  const [featured, setFeatured] = useState<Product[]>([]);
  const [newArrivals, setNewArrivals] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    async function load() {
      try {
        const [featuredRes, newRes] = await Promise.all([
          api.getProducts({ page: 1, page_size: 8, sort_by: "created", sort_order: "desc" }),
          api.getProducts({ page: 1, page_size: 4, sort_by: "created", sort_order: "desc" }),
        ]);
        setFeatured(featuredRes.items);
        setNewArrivals(newRes.items.slice(0, 4));
      } catch {
        // API might not be running yet
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/products?search=${encodeURIComponent(searchQuery.trim())}`);
    }
  }

  return (
    <div>
      {/* Hero */}
      <section className="relative bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-10 w-64 h-64 rounded-full bg-blue-400 blur-3xl" />
          <div className="absolute bottom-10 right-20 w-96 h-96 rounded-full bg-purple-400 blur-3xl" />
        </div>
        <div className="container mx-auto px-4 py-24 md:py-32 relative z-10">
          <div className="max-w-2xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 bg-blue-500/20 border border-blue-400/30 rounded-full px-4 py-1.5 text-sm mb-6">
              <Zap className="h-4 w-4 text-blue-400" />
              <span className="text-blue-200">Bộ sưu tập mới 2024</span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold leading-tight mb-6">
              Giày Chạy Bộ<br />
              <span className="text-blue-400">Chính Hãng</span> Hàng Đầu
            </h1>
            <p className="text-gray-300 text-lg mb-8">
              Khám phá danh mục thời trang từ các cửa hàng nổi bật, tìm nhanh theo mô tả, cửa hàng hoặc danh mục.
            </p>
            <form onSubmit={handleSearch} className="flex gap-2 max-w-md mx-auto mb-8">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Tìm kiếm giày..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 bg-white/10 border-white/20 text-white placeholder:text-gray-400 focus-visible:ring-blue-400"
                />
              </div>
              <Button type="submit" className="bg-blue-500 hover:bg-blue-600">Tìm</Button>
            </form>
            <div className="flex gap-4 justify-center flex-wrap">
              <Button asChild size="lg" className="bg-blue-500 hover:bg-blue-600">
                <Link to="/products">Xem Tất Cả Sản Phẩm</Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="border-white/30 text-white bg-white/10 hover:bg-white/20">
                <Link to="/products?stores=AnStore">Xem AnStore</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="bg-white border-b">
        <div className="container mx-auto px-4 py-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="flex flex-col items-center text-center gap-2">
                <div className="w-12 h-12 rounded-full bg-blue-50 flex items-center justify-center">
                  <Icon className="h-5 w-5 text-blue-600" />
                </div>
                <h3 className="font-semibold text-sm text-gray-800">{title}</h3>
                <p className="text-xs text-gray-500 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stores */}
      <section className="bg-gray-50 py-10">
        <div className="container mx-auto px-4">
          <h2 className="text-center text-sm font-semibold text-gray-400 uppercase tracking-widest mb-6">Cửa Hàng</h2>
          <div className="flex flex-wrap gap-3 justify-center">
            {STORES.map(({ name, color }) => (
              <Link
                key={name}
                to={`/products?stores=${encodeURIComponent(name)}`}
                className={`px-5 py-2 rounded-full text-sm font-semibold transition-colors ${color}`}
              >
                {name}
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* New Arrivals */}
      {newArrivals.length > 0 && (
        <section className="py-14 bg-white">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Hàng Mới Về</h2>
                <p className="text-gray-500 text-sm mt-1">Bộ sưu tập mới nhất vừa cập nhật</p>
              </div>
              <Button asChild variant="ghost" className="text-blue-600 hover:text-blue-700">
                <Link to="/products?sort_by=created&sort_order=desc" className="flex items-center gap-1">
                  Xem thêm <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {newArrivals.map((p) => <ProductCard key={p.id} product={p} />)}
            </div>
          </div>
        </section>
      )}

      {/* Featured Products */}
      <section className="py-14 bg-gray-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Sản Phẩm Nổi Bật</h2>
              <p className="text-gray-500 text-sm mt-1">Những sản phẩm mới nhất trong bộ sưu tập.</p>
            </div>
            <Button asChild variant="ghost" className="text-blue-600 hover:text-blue-700">
              <Link to="/products" className="flex items-center gap-1">
                Xem thêm <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </div>
          {loading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="bg-white rounded-xl aspect-[3/4] animate-pulse" />
              ))}
            </div>
          ) : featured.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {featured.map((p) => <ProductCard key={p.id} product={p} />)}
            </div>
          ) : (
            <div className="text-center py-16 text-gray-400">
              <p>Không có sản phẩm. Hãy chạy seed script để thêm dữ liệu mẫu.</p>
            </div>
          )}
        </div>
      </section>

      {/* CTA Banner */}
      <section className="bg-blue-600 text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-4">Sẵn Sàng Cho Cuộc Chạy Tiếp Theo?</h2>
          <p className="text-blue-100 mb-8 max-w-md mx-auto">
            Khám phá hàng trăm mẫu giày chạy bộ chính hãng với giá tốt nhất.
          </p>
          <Button asChild size="lg" className="bg-white text-blue-600 hover:bg-blue-50">
            <Link to="/products">Mua Sắm Ngay</Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
