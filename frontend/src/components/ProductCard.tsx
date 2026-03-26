import { Link } from "react-router-dom";
import { Star, Heart, MessageCircle } from "lucide-react";
import type { Product } from "@/types";
import { Badge } from "@/components/ui/badge";
import { formatPrice } from "@/lib/utils";
import { useState } from "react";

interface ProductCardProps {
  product: Product;
}

export default function ProductCard({ product }: ProductCardProps) {
  const [selectedColor, setSelectedColor] = useState(0);
  const [imgError, setImgError] = useState(false);
  const [wishlisted, setWishlisted] = useState(false);

  const imageSrc =
    !imgError && product.images.length > 0
      ? product.images[0]
      : `https://placehold.co/400x400/f3f4f6/9ca3af?text=${encodeURIComponent(product.brand)}`;

  return (
    <Link to={`/products/${product.id}`} className="group block">
      <div className="bg-white rounded-xl overflow-hidden shadow-[0_4px_20px_rgba(0,0,0,0.06)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.12)] transition-all duration-200">
        {/* Image */}
        <div className="relative aspect-square overflow-hidden bg-gray-50">
          <img
            src={imageSrc}
            alt={product.name}
            onError={() => setImgError(true)}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          />

          {/* Hover overlay */}
          <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />

          {/* Badges */}
          <div className="absolute top-2 left-2 flex flex-col gap-1">
            {product.discount_percent && (
              <Badge variant="sale" className="text-xs font-bold px-2 py-0.5 rounded">
                -{product.discount_percent}%
              </Badge>
            )}
            {product.is_new && !product.discount_percent && (
              <Badge variant="new" className="text-xs font-bold px-2 py-0.5 rounded">
                MỚI
              </Badge>
            )}
          </div>

          {/* Out of stock overlay */}
          {!product.in_stock && (
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
              <span className="text-white font-semibold text-sm bg-black/60 px-3 py-1 rounded">Hết hàng</span>
            </div>
          )}

          {/* Hover action buttons */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2 opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 transition-all duration-200">
            <button
              onClick={(e) => { e.preventDefault(); setWishlisted(!wishlisted); }}
              title="Yêu thích"
              className="w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <Heart className={`h-4 w-4 transition-colors ${wishlisted ? "fill-red-500 text-red-500" : "text-gray-500"}`} />
            </button>
            <button
              onClick={(e) => {
                e.preventDefault();
                window.dispatchEvent(new CustomEvent("add-to-chat", {
                  detail: { imageUrl: imageSrc, productName: product.name },
                }));
              }}
              title="Hỏi trợ lý về sản phẩm này"
              className="w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <MessageCircle className="h-4 w-4 text-[#015e9f]" />
            </button>
          </div>
        </div>

        {/* Info */}
        <div className="p-3 pt-[13px]">
          {/* Color swatches */}
          {product.colors.length > 0 && (
            <div className="flex gap-[6px] mb-2">
              {product.colors.slice(0, 5).map((color, i) => (
                <button
                  key={i}
                  onClick={(e) => { e.preventDefault(); setSelectedColor(i); }}
                  title={color.name}
                  className={`w-[18px] h-[18px] rounded-full border-[2px] border-white ring-1 transition-all ${
                    selectedColor === i ? "ring-[#0d1b67]" : "ring-[#d3d3d3]"
                  }`}
                  style={{ backgroundColor: color.hex }}
                />
              ))}
              {product.colors.length > 5 && (
                <span className="text-xs text-[#aeaeae] self-center">+{product.colors.length - 5}</span>
              )}
            </div>
          )}

          {/* Brand */}
          <p className="text-[13px] text-[#aeaeae] font-medium uppercase tracking-[0.04em]">{product.brand}</p>

          {/* Name */}
          <h3 className="text-[15px] font-semibold text-[#444956] mt-1.5 line-clamp-2 leading-[1.42] tracking-[-0.01em] group-hover:text-[#0d1b67] transition-colors">
            {product.name}
          </h3>

          {/* Rating */}
          {product.review_count > 0 && (
            <div className="flex items-center gap-1 mt-1">
              <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
              <span className="text-xs text-gray-400">{product.rating.toFixed(1)} ({product.review_count})</span>
            </div>
          )}

          {/* Price */}
          <div className="flex items-center gap-2 mt-2">
            <span className="text-[15px] font-semibold text-[#ff4a3a]">{formatPrice(product.price)}</span>
            {product.original_price && product.original_price > product.price && (
              <span className="text-[13px] text-[#a7a7a7] line-through">{formatPrice(product.original_price)}</span>
            )}
          </div>
        </div>
      </div>
    </Link>
  );
}
