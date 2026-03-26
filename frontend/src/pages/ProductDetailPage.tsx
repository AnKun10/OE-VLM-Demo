import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ChevronRight, Star, Truck, RotateCcw, Shield, ShoppingBag, Heart, ChevronLeft, ChevronRight as ChevronRightIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import ProductCard from "@/components/ProductCard";
import { api } from "@/lib/api";
import type { Product } from "@/types";
import { formatPrice, cn } from "@/lib/utils";

export default function ProductDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [product, setProduct] = useState<Product | null>(null);
  const [related, setRelated] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedColor, setSelectedColor] = useState(0);
  const [selectedSize, setSelectedSize] = useState<string | null>(null);
  const [quantity, setQuantity] = useState(1);
  const [activeImage, setActiveImage] = useState(0);
  const [imgError, setImgError] = useState<Record<number, boolean>>({});
  const [addedToCart, setAddedToCart] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setSelectedColor(0);
    setSelectedSize(null);
    setActiveImage(0);
    setImgError({});

    Promise.all([api.getProduct(id), api.getRelatedProducts(id, 4)])
      .then(([prod, rel]) => {
        setProduct(prod);
        setRelated(rel);
      })
      .catch(() => setProduct(null))
      .finally(() => setLoading(false));
  }, [id]);

  function handleAddToCart() {
    if (!selectedSize) return;
    setAddedToCart(true);
    setTimeout(() => setAddedToCart(false), 2000);
  }

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-10">
        <div className="grid md:grid-cols-2 gap-10 animate-pulse">
          <div className="aspect-square bg-gray-200 rounded-2xl" />
          <div className="space-y-4">
            <div className="h-6 bg-gray-200 rounded w-1/3" />
            <div className="h-8 bg-gray-200 rounded w-3/4" />
            <div className="h-10 bg-gray-200 rounded w-1/2" />
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

  const images = product.images.length > 0 ? product.images : [""];
  const activeImgSrc =
    !imgError[activeImage] && images[activeImage]
      ? images[activeImage]
      : `https://placehold.co/600x600/f3f4f6/9ca3af?text=${encodeURIComponent(product.brand)}`;

  return (
    <div className="bg-white">
      <div className="container mx-auto px-4 py-6">
        {/* Breadcrumb */}
        <nav className="flex items-center gap-1 text-sm text-gray-500 mb-6">
          <Link to="/" className="hover:text-gray-700">Trang chủ</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <Link to="/products" className="hover:text-gray-700">Sản phẩm</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <Link to={`/products?brands=${product.brand}`} className="hover:text-gray-700">{product.brand}</Link>
          <ChevronRight className="h-3.5 w-3.5" />
          <span className="text-gray-700 truncate max-w-[200px]">{product.name}</span>
        </nav>

        <div className="grid md:grid-cols-2 gap-8 lg:gap-12">
          {/* Image Gallery */}
          <div className="space-y-3">
            {/* Main image */}
            <div className="relative aspect-square bg-gray-50 rounded-2xl overflow-hidden group">
              <img
                src={activeImgSrc}
                alt={product.name}
                onError={() => setImgError((prev) => ({ ...prev, [activeImage]: true }))}
                className="w-full h-full object-cover"
              />
              {product.discount_percent && (
                <Badge variant="sale" className="absolute top-4 left-4 text-sm px-3 py-1">
                  -{product.discount_percent}%
                </Badge>
              )}
              {product.is_new && !product.discount_percent && (
                <Badge variant="new" className="absolute top-4 left-4 text-sm px-3 py-1">MỚI</Badge>
              )}
              {/* Prev/Next */}
              {images.length > 1 && (
                <>
                  <button
                    onClick={() => setActiveImage((i) => (i - 1 + images.length) % images.length)}
                    className="absolute left-3 top-1/2 -translate-y-1/2 w-9 h-9 bg-white/80 rounded-full flex items-center justify-center shadow opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <ChevronLeft className="h-5 w-5" />
                  </button>
                  <button
                    onClick={() => setActiveImage((i) => (i + 1) % images.length)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 bg-white/80 rounded-full flex items-center justify-center shadow opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <ChevronRightIcon className="h-5 w-5" />
                  </button>
                </>
              )}
            </div>
            {/* Thumbnails */}
            {images.length > 1 && (
              <div className="flex gap-2 overflow-x-auto pb-1">
                {images.map((src, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveImage(i)}
                    className={cn(
                      "w-16 h-16 rounded-lg overflow-hidden border-2 shrink-0 transition-colors",
                      activeImage === i ? "border-blue-500" : "border-gray-200 hover:border-gray-400"
                    )}
                  >
                    <img
                      src={imgError[i] ? `https://placehold.co/64x64/f3f4f6/9ca3af?text=${i + 1}` : src}
                      alt={`${product.name} ${i + 1}`}
                      onError={() => setImgError((prev) => ({ ...prev, [i]: true }))}
                      className="w-full h-full object-cover"
                    />
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Product Info */}
          <div className="flex flex-col gap-4">
            {/* Brand & Name */}
            <div>
              <Link
                to={`/products?brands=${product.brand}`}
                className="text-sm font-semibold text-blue-600 hover:text-blue-700 uppercase tracking-wide"
              >
                {product.brand}
              </Link>
              <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mt-1 leading-tight">{product.name}</h1>
            </div>

            {/* Rating */}
            {product.review_count > 0 && (
              <div className="flex items-center gap-2">
                <div className="flex">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Star
                      key={i}
                      className={cn(
                        "h-4 w-4",
                        i < Math.floor(product.rating) ? "fill-yellow-400 text-yellow-400" : "text-gray-300"
                      )}
                    />
                  ))}
                </div>
                <span className="text-sm font-medium text-gray-700">{product.rating.toFixed(1)}</span>
                <span className="text-sm text-gray-400">({product.review_count} đánh giá)</span>
              </div>
            )}

            {/* Price */}
            <div className="flex items-end gap-3">
              <span className="text-3xl font-bold text-red-500">{formatPrice(product.price)}</span>
              {product.original_price && product.original_price > product.price && (
                <>
                  <span className="text-lg text-gray-400 line-through">{formatPrice(product.original_price)}</span>
                  <Badge variant="sale">-{product.discount_percent}%</Badge>
                </>
              )}
            </div>

            {/* Stock */}
            <div className="flex items-center gap-2">
              <div className={cn("w-2 h-2 rounded-full", product.in_stock ? "bg-green-500" : "bg-red-500")} />
              <span className="text-sm text-gray-600">
                {product.in_stock ? `Còn hàng (${product.stock_qty} đôi)` : "Hết hàng"}
              </span>
            </div>

            <Separator />

            {/* Colors */}
            {product.colors.length > 0 && (
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-2">
                  Màu sắc: <span className="font-normal text-gray-500">{product.colors[selectedColor]?.name}</span>
                </p>
                <div className="flex gap-2 flex-wrap">
                  {product.colors.map((color, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedColor(i)}
                      title={color.name}
                      className={cn(
                        "w-8 h-8 rounded-full border-2 transition-all",
                        selectedColor === i ? "border-gray-800 scale-110 shadow-md" : "border-gray-300 hover:border-gray-500"
                      )}
                      style={{ backgroundColor: color.hex }}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Sizes */}
            {product.sizes.length > 0 && (
              <div>
                <p className="text-sm font-semibold text-gray-700 mb-2">
                  Kích cỡ: {selectedSize && <span className="font-normal text-gray-500">{selectedSize}</span>}
                  {!selectedSize && <span className="font-normal text-red-400">* Vui lòng chọn size</span>}
                </p>
                <div className="flex gap-2 flex-wrap">
                  {product.sizes.map((size) => (
                    <button
                      key={size}
                      onClick={() => setSelectedSize(size)}
                      className={cn(
                        "px-4 py-2 text-sm border rounded-lg font-medium transition-colors",
                        selectedSize === size
                          ? "bg-gray-900 text-white border-gray-900"
                          : "bg-white text-gray-700 border-gray-300 hover:border-gray-600"
                      )}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Quantity */}
            <div className="flex items-center gap-3">
              <p className="text-sm font-semibold text-gray-700">Số lượng:</p>
              <div className="flex items-center border rounded-lg overflow-hidden">
                <button
                  onClick={() => setQuantity((q) => Math.max(1, q - 1))}
                  className="px-3 py-2 hover:bg-gray-50 text-gray-600 transition-colors"
                >-</button>
                <span className="px-4 py-2 text-sm font-medium border-x">{quantity}</span>
                <button
                  onClick={() => setQuantity((q) => Math.min(product.stock_qty, q + 1))}
                  className="px-3 py-2 hover:bg-gray-50 text-gray-600 transition-colors"
                >+</button>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 flex-col sm:flex-row">
              <Button
                onClick={handleAddToCart}
                disabled={!product.in_stock || !selectedSize}
                className={cn(
                  "flex-1 gap-2 transition-all",
                  addedToCart ? "bg-green-600 hover:bg-green-700" : "bg-blue-600 hover:bg-blue-700"
                )}
                size="lg"
              >
                <ShoppingBag className="h-5 w-5" />
                {addedToCart ? "Đã thêm vào giỏ!" : "Thêm vào giỏ hàng"}
              </Button>
              <Button variant="outline" size="lg" className="sm:w-auto">
                <Heart className="h-5 w-5" />
              </Button>
            </div>

            {/* Benefits */}
            <div className="grid grid-cols-3 gap-3 mt-2">
              {[
                { icon: Truck, text: "Giao toàn quốc" },
                { icon: RotateCcw, text: "Đổi trả 30 ngày" },
                { icon: Shield, text: "Hàng chính hãng" },
              ].map(({ icon: Icon, text }) => (
                <div key={text} className="flex flex-col items-center gap-1 p-3 bg-gray-50 rounded-xl text-center">
                  <Icon className="h-5 w-5 text-blue-600" />
                  <span className="text-xs text-gray-600">{text}</span>
                </div>
              ))}
            </div>

            <Separator />

            {/* Description */}
            <div>
              <h3 className="font-semibold text-gray-800 mb-2">Mô tả sản phẩm</h3>
              <p className="text-sm text-gray-600 leading-relaxed">{product.description}</p>
            </div>

            {/* Tags */}
            {product.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {product.tags.map((tag) => (
                  <Link
                    key={tag}
                    to={`/products?search=${encodeURIComponent(tag)}`}
                    className="text-xs bg-gray-100 text-gray-500 px-2.5 py-1 rounded-full hover:bg-gray-200 transition-colors"
                  >
                    #{tag}
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Related Products */}
        {related.length > 0 && (
          <div className="mt-16">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Sản Phẩm Liên Quan</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {related.map((p) => <ProductCard key={p.id} product={p} />)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
