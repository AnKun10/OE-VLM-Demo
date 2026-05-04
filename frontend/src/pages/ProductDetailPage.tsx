import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import ProductCard from "@/components/ProductCard";
import { api } from "@/lib/api";
import type { Product } from "@/types";

export default function ProductDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [product, setProduct] = useState<Product | null>(null);
  const [related, setRelated] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [imgError, setImgError] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setImgError(false);

    Promise.all([api.getProduct(id), api.getRelatedProducts(id, 4)])
      .then(([prod, rel]) => {
        setProduct(prod);
        setRelated(rel);
      })
      .catch(() => setProduct(null))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-10">
        <div className="grid md:grid-cols-2 gap-10 animate-pulse">
          <div className="aspect-square bg-gray-200 rounded-2xl" />
          <div className="space-y-4">
            <div className="h-6 bg-gray-200 rounded w-1/3" />
            <div className="h-8 bg-gray-200 rounded w-3/4" />
            <div className="h-24 bg-gray-200 rounded" />
          </div>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="container mx-auto px-4 py-20 text-center">
        <div className="text-6xl mb-4">😕</div>
        <h2 className="text-xl font-bold text-gray-700 mb-2">Không tìm thấy sản phẩm</h2>
        <p className="text-gray-400 mb-6">Sản phẩm này không tồn tại hoặc đã bị xóa.</p>
        <Button asChild><Link to="/products">Quay lại danh sách</Link></Button>
      </div>
    );
  }

  const imageSrc = !imgError && product.image_url
    ? product.image_url
    : "https://placehold.co/800x800?text=No+Image";

  return (
    <div className="bg-white">
      <div className="container mx-auto px-4 py-6">
        <nav className="flex items-center gap-1 text-sm text-gray-500 mb-6">
          <Link to="/" className="hover:text-gray-700">Trang chủ</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <Link to="/products" className="hover:text-gray-700">Sản phẩm</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <Link to={`/products?stores=${encodeURIComponent(product.store)}`} className="hover:text-gray-700">{product.store}</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <span className="text-gray-700 truncate max-w-[200px]">{product.name}</span>
        </nav>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          <div className="aspect-square bg-gray-50 rounded-2xl overflow-hidden">
            <img
              src={imageSrc}
              alt={product.name}
              onError={() => setImgError(true)}
              className="w-full h-full object-cover"
            />
          </div>

          <div className="flex flex-col gap-5">
            <div>
              <Link
                to={`/products?stores=${encodeURIComponent(product.store)}`}
                className="text-sm font-semibold text-blue-600 hover:text-blue-700 uppercase tracking-wide"
              >
                {product.store}
              </Link>
              <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mt-1 leading-tight">{product.name}</h1>
            </div>

            <div className="flex flex-wrap gap-2 text-sm">
              <Link
                to={`/products?categories=${encodeURIComponent(product.category)}`}
                className="rounded-full bg-gray-100 px-3 py-1 text-gray-700 hover:bg-gray-200"
              >
                {product.category}
              </Link>
            </div>

            <div className="pt-2">
              <Button asChild>
                <Link to="/products">Quay lại danh sách</Link>
              </Button>
            </div>
          </div>
        </div>

        {related.length > 0 && (
          <section className="mt-14">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Sản phẩm liên quan</h2>
                <p className="text-sm text-gray-500 mt-1">Cùng nhóm sản phẩm hoặc cửa hàng.</p>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
              {related.map((item) => <ProductCard key={item.id} product={item} />)}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
