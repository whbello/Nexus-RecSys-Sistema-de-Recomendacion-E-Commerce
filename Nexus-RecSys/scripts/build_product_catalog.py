"""
build_product_catalog.py
========================
Construye el catálogo de productos para la demo final de nexus-recsys.

Estrategia:
- Top 500 ítems por popularidad → OPCIÓN A: mapeo a productos reales de e-commerce
- Resto del catálogo          → OPCIÓN B: enriquecer con categoría del dataset

Ejecutar: python scripts/build_product_catalog.py
Output:   data/processed/product_catalog.json
"""

import json
import os
import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
OUTPUT_PATH = DATA_PROC / "product_catalog.json"

RANDOM_SEED = 42
TOP_N       = 500  # ítems de Opción A

# ── Mapeo de categorías raíz numéricas → nombres de texto ─────────────────────
# (category_tree.csv solo tiene IDs numéricos — se asignan nombres para la demo)
ROOT_CATEGORY_IDS = [231, 791, 1490, 431, 755, 378, 1579, 1394, 659, 1057,
                     859, 803, 250, 1452, 1182, 1692, 1600, 1482, 1224, 1532,
                     395, 653, 140, 1698, 679]

# Mapeo demo: IDs numéricos → etiquetas de texto
ROOT_CAT_NAMES = {
    231:  "Electrónica",
    791:  "Ropa y Moda",
    1490: "Calzado",
    431:  "Hogar y Decoración",
    755:  "Deportes y Fitness",
    378:  "Libros y Educación",
    1579: "Belleza y Cuidado Personal",
    1394: "Alimentos y Bebidas",
    659:  "Juguetes y Entretenimiento",
    1057: "Herramientas y Bricolaje",
    859:  "Salud y Farmacia",
    803:  "Automóviles y Motos",
    250:  "Jardín y Exterior",
    1452: "Mascotas",
    1182: "Arte y Manualidades",
    1692: "Música e Instrumentos",
    1600: "Oficina y Papelería",
    1482: "Viajes y Equipaje",
    1224: "Bebés y Maternidad",
    1532: "Accesorios y Joyería",
    395:  "Coleccionables",
    653:  "Electrónica de Consumo",
    140:  "Software y Videojuegos",
    1698: "Servicios Digitales",
    679:  "Otros",
}

# Emojis por categoría para Opción B
CATEGORY_EMOJIS = {
    "Electrónica":              "💻",
    "Electrónica de Consumo":   "📱",
    "Ropa y Moda":              "👕",
    "Calzado":                  "👟",
    "Hogar y Decoración":       "🏠",
    "Deportes y Fitness":       "⚽",
    "Libros y Educación":       "📚",
    "Belleza y Cuidado Personal": "💄",
    "Alimentos y Bebidas":      "🍎",
    "Juguetes y Entretenimiento": "🎮",
    "Herramientas y Bricolaje": "🔧",
    "Salud y Farmacia":         "💊",
    "Automóviles y Motos":      "🚗",
    "Jardín y Exterior":        "🌿",
    "Mascotas":                 "🐾",
    "Arte y Manualidades":      "🎨",
    "Música e Instrumentos":    "🎵",
    "Oficina y Papelería":      "🖊️",
    "Viajes y Equipaje":        "✈️",
    "Bebés y Maternidad":       "👶",
    "Accesorios y Joyería":     "💍",
    "Coleccionables":           "🏆",
    "Software y Videojuegos":   "🕹️",
    "Servicios Digitales":      "☁️",
    "Otros":                    "📦",
    "default":                  "📦",
}

# ── Catálogo de productos reales (Opción A) ────────────────────────────────────
# Distribuidos proporcionalmente según las categorías del dataset:
#   Electrónica (25%), Ropa y Calzado (20%), Hogar (15%), Deportes (15%),
#   Libros (10%), Belleza (10%), Alimentos (5%)
OPCION_A_PRODUCTOS = [
    # ── ELECTRÓNICA (25% = 125 productos) ─────────────────────────────────────
    ("Auriculares Bluetooth Sony WH-1000XM5",        "Electrónica", "Audio",             299.0, "🎧",
     "Cancelación de ruido líder en la industria, 30h de batería"),
    ("Smartphone Samsung Galaxy A54 5G",             "Electrónica", "Smartphones",        449.0, "📱",
     "Pantalla AMOLED 6.4\", 5000 mAh, triple cámara 50 MP"),
    ("Laptop ASUS VivoBook 15",                      "Electrónica", "Computadoras",       799.0, "💻",
     "Core i5-12500H, 8 GB RAM, SSD 512 GB, pantalla FHD IPS"),
    ("Tablet iPad Air 5ta Gen",                      "Electrónica", "Tablets",            749.0, "📱",
     "Chip M1, pantalla Liquid Retina 10.9\", compatible con Pencil 2"),
    ("Smartwatch Xiaomi Mi Watch",                   "Electrónica", "Wearables",           89.0, "⌚",
     "GPS integrado, 117 modos deportivos, 16 días de batería"),
    ("Cámara GoPro Hero 12 Black",                   "Electrónica", "Fotografía",         399.0, "📷",
     "Video 5.3K · HyperSmooth 6.0 · waterproof 10m"),
    ("Monitor LG 27\" 4K UHD",                       "Electrónica", "Monitores",          349.0, "🖥️",
     "Panel IPS, 144 Hz, HDR400, USB-C, compatible con PS5"),
    ("Teclado Mecánico Logitech MX Keys",            "Electrónica", "Periféricos",        119.0, "⌨️",
     "Retroiluminación inteligente, multi-device, recargable"),
    ("Mouse Inalámbrico Logitech MX Master 3S",      "Electrónica", "Periféricos",         99.0, "🖱️",
     "8000 DPI, silencioso, 70 días de batería, USB-C"),
    ("Parlante JBL Charge 5",                        "Electrónica", "Audio",              179.0, "🔊",
     "20h de batería, carga otros dispositivos, resistente al agua IP67"),
    ("Auriculares In-Ear Apple AirPods Pro 2da Gen", "Electrónica", "Audio",              249.0, "🎵",
     "ANC adaptativa, chip H2, resistencia al agua IPX4"),
    ("Router WiFi 6 TP-Link Archer AX73",            "Electrónica", "Redes",              149.0, "📡",
     "WiFi 6, AX5400, 4×4 MU-MIMO, cobertura hasta 250 m²"),
    ("SSD Externo Samsung T7 Shield 2TB",            "Electrónica", "Almacenamiento",     129.0, "💾",
     "1050 MB/s, USB 3.2 Gen2, resistente a golpes IP65"),
    ("Cargador Inalámbrico Anker 15W",               "Electrónica", "Accesorios",          29.0, "⚡",
     "Qi universal, compatible con MagSafe, LED indicador"),
    ("Power Bank Anker 20000mAh",                    "Electrónica", "Accesorios",          49.0, "🔋",
     "Carga rápida 20W PD3.0, 2 puertos USB-A + 1 USB-C, compacto"),
    ("Smart TV Samsung QLED 55\" 4K",                "Electrónica", "TV y Video",         699.0, "📺",
     "Quantum Dot, 120 Hz, Tizen OS, HDMI 2.1 para gaming"),
    ("Consola PlayStation 5 Slim",                   "Electrónica", "Gaming",             449.0, "🕹️",
     "SSD 1TB, ray tracing, 120 fps, retrocompatible PS4"),
    ("Drone DJI Mini 4 Pro",                         "Electrónica", "Fotografía",         759.0, "🚁",
     "4K/60fps, obstáculos omnidireccionales, 34 min de vuelo"),
    ("Proyector Epson Home Cinema 2350",             "Electrónica", "TV y Video",         899.0, "📽️",
     "Full HD 2800 lúmenes, 3LCD, 3D, Bluetooth audio integrado"),
    ("Hub USB-C 12-en-1 Anker",                      "Electrónica", "Accesorios",          59.0, "🔌",
     "4K HDMI, SD card, 100W PD, USB 3.0 ×4, Ethernet 1Gbps"),
    ("Teclado Gaming Razer BlackWidow V4",           "Electrónica", "Gaming",             139.0, "⌨️",
     "Switches mecánicos Green, anti-ghosting, chroma RGB"),
    ("Auriculares Gaming HyperX Cloud Alpha",        "Electrónica", "Gaming",              99.0, "🎮",
     "Drivers dobles, cancelación de ruido, micrófono QD"),
    ("Cámara de Seguridad Arlo Pro 5S",              "Electrónica", "Seguridad",          199.0, "📹",
     "2K HDR, color a noche, sirena integrada, sin suscripción"),
    ("Tableta Gráfica Wacom Intuos M",               "Electrónica", "Accesorios",          79.0, "✏️",
     "Área activa 21.6×13.5 cm, 4096 niveles presión, USB"),
    ("Impresora HP LaserJet M140w",                  "Electrónica", "Impresión",          149.0, "🖨️",
     "Mono láser, WiFi, 21 ppm, bandeja 150 hojas"),
    ("Altavoz Inteligente Amazon Echo Dot 5",        "Electrónica", "Smart Home",          49.0, "🔊",
     "Alexa integrada, sonido mejorado, hub Zigbee incorporado"),
    ("Cargador Solar 21W Anker PowerPort Solar",     "Electrónica", "Energía",             59.0, "☀️",
     "2 puertos USB, plegable, IQ carga inteligente, IPX3"),
    ("Cable HDMI 2.1 Belkin Ultra HD 2m",            "Electrónica", "Cables",              19.0, "🔌",
     "8K@60Hz, 4K@120Hz, 48Gbps, compatible PS5/Xbox"),
    ("Micrófono Podcast Blue Yeti USB",              "Electrónica", "Audio",              129.0, "🎙️",
     "4 patrones polares, filtro pop integrado, gain control"),
    ("Webcam Logitech C920 Pro HD",                  "Electrónica", "Periféricos",         79.0, "📷",
     "1080p/30fps, autofoco, corrección de luz automática"),
    ("Lámpara LED de Escritorio BenQ e-Reading",    "Electrónica", "Iluminación",        149.0, "💡",
     "Sin glare, USB-C, ajuste de temperatura de color automático"),
    ("Disco Duro Externo WD My Passport 4TB",        "Electrónica", "Almacenamiento",      89.0, "💽",
     "USB 3.0, cifrado AES-256, respaldo automático, 3 años garantía"),
    ("Adaptador WiFi USB ASUS USB-AX56",             "Electrónica", "Redes",               44.0, "📡",
     "WiFi 6 AX1800, MU-MIMO, USB 3.0, hasta 1800 Mbps"),
    ("Placa Base AMD B650 ASUS ROG Strix",           "Electrónica", "PC Gaming",          249.0, "🖥️",
     "AM5, PCIe 5.0, DDR5, WiFi 6E integrado, 2.5GbE"),
    ("Tarjeta Gráfica NVIDIA GeForce RTX 4060",      "Electrónica", "PC Gaming",          299.0, "🖥️",
     "8 GB GDDR6, DLSS 3, ray tracing, AV1 encode"),
    ("Procesador AMD Ryzen 5 7600X",                 "Electrónica", "Componentes PC",     229.0, "⚙️",
     "6 núcleos / 12 hilos, 5.3 GHz boost, AM5, 105W TDP"),
    ("RAM DDR5 Corsair Vengeance 32GB 6000MHz",      "Electrónica", "Componentes PC",     119.0, "💾",
     "Kit 2×16 GB, XMP 3.0, CL36, doble canal"),
    ("Pantalla Portátil ASUS ZenScreen 15.6\" FHD",  "Electrónica", "Monitores",          199.0, "🖥️",
     "USB-C, 60 Hz, peso 0.78 kg, auto-rotación Smart Case"),
    ("Sticks HDMI Amazon Fire TV 4K Max",            "Electrónica", "Streaming",           59.0, "📺",
     "4K 60fps, Dolby Vision/Atmos, WiFi 6, Alexa integrada"),
    ("Generador Portátil Jackery Solar 1000 Pro",    "Electrónica", "Energía",           1099.0, "⚡",
     "1002 Wh, carga solar 100W, AC 1000W, silencioso"),
    ("Controlador Midi Arturia MiniLab 3",           "Electrónica", "Música",              89.0, "🎹",
     "25 teclas mini, 8 pads, 9 faders, Ableton incluido"),
    ("Smart Lock Yale Assure SL",                    "Electrónica", "Smart Home",         199.0, "🔒",
     "Sin llave, touchscreen, compatible Apple HomeKey y Alexa"),
    ("Termostato Inteligente Nest Learning 4ta gen", "Electrónica", "Smart Home",         249.0, "🌡️",
     "Aprendizaje automático, HVAC universal, ENERGY STAR"),
    ("Aspiradora Robot iRobot Roomba j9+",           "Electrónica", "Electrodomésticos",  799.0, "🤖",
     "Vaciado automático, evita cables y mascotas, mapa de casa"),
    ("Escáner Portátil Fujitsu ScanSnap iX1600",     "Electrónica", "Oficina",            399.0, "📄",
     "40 ppm, WiFi, cloud directo, pantalla táctil, doble cara"),
    # ── Subtotal Electrónica: 45 productos ─────────────────────────────────────

    # ── ROPA Y MODA (20% = 100 productos) ──────────────────────────────────────
    ("Zapatillas Nike Air Max 270",                  "Ropa y Moda", "Calzado Deportivo",  129.0, "👟",
     "Unidad Air Max de 270° en el talón, foam ultrasuave"),
    ("Zapatillas Adidas Ultraboost 23",              "Ropa y Moda", "Calzado Deportivo",  159.0, "👟",
     "Boost de 3 capas, soporte de arco Primeknit, sostenible"),
    ("Campera North Face Thermoball Eco",            "Ropa y Moda", "Abrigos",            199.0, "🧥",
     "Plumón sintético, compresible, recycled, DWR finish"),
    ("Remera Under Armour UA Tech",                  "Ropa y Moda", "Camisetas",           35.0, "👕",
     "HeatGear, UPF 30, transpirable, secado rápido"),
    ("Jeans Levi's 501 Original Fit",                "Ropa y Moda", "Pantalones",          89.0, "👖",
     "Algodón 100%, corte recto clásico, botones de metal"),
    ("Buzo Adidas Essentials Fleece",                "Ropa y Moda", "Buzos",               49.0, "👕",
     "Fleece de algodón 70%, canguro, logo Trefoil bordado"),
    ("Zapatillas Converse Chuck Taylor All Star OX", "Ropa y Moda", "Calzado Casual",      65.0, "👟",
     "Lona duradera, puntera de goma, icono cultural desde 1917"),
    ("Vestido Zara Satinado Midi",                   "Ropa y Moda", "Vestidos",            79.0, "👗",
     "Satén con tirantes finos, largo midi, escote V"),
    ("Campera Impermeable Columbia Arcadia II",      "Ropa y Moda", "Impermeables",        89.0, "🧥",
     "Omni-Tech impermeable, costuras selladas, empaquetable"),
    ("Zapatillas New Balance 574",                   "Ropa y Moda", "Calzado Casual",      89.0, "👟",
     "Suela ENCAP, suede/malla, retro lifestyle"),
    ("Gorro Estilo Bucket Hat New Era",              "Ropa y Moda", "Accesorios",          29.0, "🧢",
     "Algodón 100%, reversible, talla ajustable"),
    ("Bufanda Merino Wool Uniqlo",                   "Ropa y Moda", "Accesorios",          39.0, "🧣",
     "Lana merina 100%, suave al tacto, 180×25 cm"),
    ("Medias Compresión CEP Run 3.0",                "Ropa y Moda", "Ropa Deportiva",      39.0, "🧦",
     "15–20 mmHg, antibacteriano, secado ultrarrápido"),
    ("Zapatillas Vans Old Skool",                    "Ropa y Moda", "Calzado Casual",      70.0, "👟",
     "Sidestripe icónico, suela waffle, skate classic"),
    ("Polo Lacoste Regular Fit",                     "Ropa y Moda", "Camisetas",           89.0, "👕",
     "Piqué petit de algodón, cocodrilo bordado, 25 colores"),
    ("Corset Bershka Denim",                         "Ropa y Moda", "Tops",                39.0, "👙",
     "Denim elástico, cierre de ojales, tirantes regulables"),
    ("Pantalón Yoga Lululemon Align 25\"",            "Ropa y Moda", "Ropa Deportiva",     128.0, "🧘",
     "Nulu fabric, sin bolsillos frontales, sensación de nada"),
    ("Zapatillas ASICS Gel-Nimbus 25",               "Ropa y Moda", "Calzado Running",    160.0, "👟",
     "Amortiguación FF Blast PLUS ECO, OrthoLite 3.0"),
    ("Musculosa Nike Dri-FIT",                       "Ropa y Moda", "Camisetas",           29.0, "👕",
     "Dri-FIT, espalda con malla, moldeo ergonómico"),
    ("Short Nike Challenger 7\"",                    "Ropa y Moda", "Pantalones",          35.0, "🩳",
     "Bolsillo trasero, forro interior, corte Dri-FIT"),
    ("Calzado Oxford Tom Ford Cap Toe",              "Ropa y Moda", "Calzado Formal",     890.0, "👞",
     "Cuero de becerro italiana, encaje dorado, suela Blake"),
    ("Hoodie Champion Reverse Weave",                "Ropa y Moda", "Sudaderas",           75.0, "👕",
     "Fleece reverse weave, canguro, ribeteo en contraste"),
    ("Chaqueta Levi's Trucker Denim",                "Ropa y Moda", "Abrigos",             89.0, "🧥",
     "Denim rígido 13 oz, cierre de botones, logo parche"),
    ("Zapatillas Reebok Classic Leather",            "Ropa y Moda", "Calzado Casual",      74.0, "👟",
     "Cuero genuino, suela EVA, comfort clásico desde 1983"),
    ("Mallas Compresión 2XU Mid-Rise",               "Ropa y Moda", "Ropa Deportiva",      89.0, "🧘",
     "Tejido ICE X, 50+ UPF, leve compresión gradiente"),
    ("Camisa Oxford Brooks Brothers",                "Ropa y Moda", "Camisas",             98.0, "👔",
     "Algodón Supima, cuello button-down, no iron tratada"),
    ("Zapatillas Puma RS-X",                         "Ropa y Moda", "Calzado Casual",      99.0, "👟",
     "Foam RS espeso, malla chunky, colores loud"),
    ("Traje Slim H&M Premium",                       "Ropa y Moda", "Ropa Formal",        199.0, "🤵",
     "Mezcla lana, slim fit, solapa pico, forro completo"),
    ("Bota Dr. Martens 1460 Pascal",                 "Ropa y Moda", "Calzado",            149.0, "🥾",
     "Cuero Virginia, 8 ojales, suela AirWair bouncing"),
    ("Falda Midi Plisada Zara",                      "Ropa y Moda", "Faldas",              49.0, "👗",
     "Plisado satinado, elástico en cintura, largo midi"),
    ("Gafas de Sol Ray-Ban Aviator Classic",         "Ropa y Moda", "Accesorios",         159.0, "😎",
     "Cristal mineral, montura metal, UV400, polarizado disponible"),
    ("Bolso de Mano Guess Factory Logo",             "Ropa y Moda", "Bolsos",              79.0, "👜",
     "PU con logo all-over, asa de mano, bandolera extraíble"),
    ("Cinturón Tommy Hilfiger Eton 3.5 cm",         "Ropa y Moda", "Accesorios",          49.0, "👔",
     "Cuero genuino, hebilla logotipo, reversible"),
    ("Guantes Invernales Thinsulate Trespass",       "Ropa y Moda", "Accesorios",          24.0, "🧤",
     "Thinsulate 40g, pantalla táctil, elastico en muñeca"),
    ("Traje de Baño Speedo Fastskin",               "Ropa y Moda", "Natación",             89.0, "🏊",
     "Tejido ultrafino FLEX, costura de alta gama, UPF50+"),
    ("Corbata Slim Tie Calvin Klein",                "Ropa y Moda", "Accesorios",          45.0, "👔",
     "Seda 100%, ancho 6 cm, varios colores sólidos"),
    ("Mochila Tipo Escuela Herschel Little America", "Ropa y Moda", "Mochilas",            99.0, "🎒",
     "Compartimento laptop 15\", vintage straps, 25 L"),
    ("Maletín Business Samsonite Openroad Chic",    "Ropa y Moda", "Bolsos",              139.0, "💼",
     "Asa de aluminio, 3 compartimentos, USB port"),
    ("Boina de Lana Beret Stetson",                  "Ropa y Moda", "Accesorios",          39.0, "🎩",
     "Lana pura, talla única, 10 colores"),
    ("Chaleco Acolchado The North Face Aconcagua 3", "Ropa y Moda", "Chalecos",            149.0, "🧥",
     "Down 550 fill, sin mangas, cierre YKK, empaquetable"),
    ("Zapatillas Salomon Speedcross 6",              "Ropa y Moda", "Trail Running",       139.0, "👟",
     "Agarre Contagrip, DROP 10mm, Quicklace + sistema prolock"),
    ("Jersey Lana Merino Uniqlo Extra Fine",         "Ropa y Moda", "Sweaters",            59.0, "🧥",
     "Merino Australian Grade A, cuello redondo, colores neutros"),
    ("Bermuda Cargo Carhartt Rigby",                 "Ropa y Moda", "Pantalones Cortos",   79.0, "🩳",
     "Canvas 8.5 oz, 8 bolsillos, dobladillo original"),
    ("Calcetines Pack 7 días Tommy Hilfiger",        "Ropa y Moda", "Calcetines",          39.0, "🧦",
     "Algodón peinado 80%, talla única 39-46"),
    ("Tank Top Gym Gymshark Vital",                  "Ropa y Moda", "Ropa Deportiva",      34.0, "💪",
     "DWR Speedfit fabric, corte oversized, logo reflectante"),
    # Subtotal Ropa: 45 productos ───────────────────────────────────────────────

    # ── HOGAR Y DECORACIÓN (15% = 75 productos) ────────────────────────────────
    ("Cafetera Nespresso Vertuo Pop",                "Hogar y Decoración", "Cafeteras",    149.0, "☕",
     "Cápsulas Vertuo, centrifusión, 5 tamaños de taza, autoapagado"),
    ("Silla Gamer DXRacer Formula Series",           "Hogar y Decoración", "Muebles",      329.0, "🪑",
     "PVC cuero, altura ajustable, reposacabezas y lumbar incluidos"),
    ("Lámpara Smart Philips Hue White Ambiance",     "Hogar y Decoración", "Iluminación",   59.0, "💡",
     "E27 LED, 2700-6500K, control app/Alexa, programación"),
    ("Almohada Viscoelástica Premium Tempur",        "Hogar y Decoración", "Dormitorio",    79.0, "🛏️",
     "Material TEMPUR original, ErgoNeck o Cloud disponibles"),
    ("Juego de Sábanas 400 hilos Brooklinen Luxe",  "Hogar y Decoración", "Dormitorio",   149.0, "🛏️",
     "Percal 100% algodón largo-staple, OEKO-TEX, 6 colores"),
    ("Colchón Casper Wave Hybrid King",              "Hogar y Decoración", "Dormitorio",   2295.0,"🛏️",
     "Gel pods, 5 zonas de soporte, spring híbrido, 25 cm"),
    ("Sofá 3 Plazas IKEA KIVIK",                    "Hogar y Decoración", "Muebles",       699.0, "🛋️",
     "Estructura madera maciza, funda lavable, múltiples colores"),
    ("Mesa de Centro Madera de Roble Nielsen",       "Hogar y Decoración", "Muebles",      299.0, "🪵",
     "Roble macizo danés, acabado al agua, 120×60 cm"),
    ("Set de Cuchillos de Chef Wüsthof Classic 7pc", "Hogar y Decoración", "Cocina",       349.0, "🔪",
     "Acero inox X50CrMoV15, forjado a mano en Solingen"),
    ("Batería de Cocina All-Clad D3 10 piezas",     "Hogar y Decoración", "Cocina",        699.0, "🍳",
     "Acero inox tri-ply, compatible inducción, USA made"),
    ("Aerogrill Ninja Foodi 9-in-1",                "Hogar y Decoración", "Electrodomésticos", 199.0, "🍽️",
     "Freidora, grill, asador, deshidratador, 5.7L"),
    ("Robot de Cocina Thermomix TM6",               "Hogar y Decoración", "Electrodomésticos", 1399.0,"🥘",
     "22 funciones, 3000 recetas integradas, WiFi, pantalla 6\""),
    ("Cafetera Espresso De'Longhi Dinamica Plus",   "Hogar y Decoración", "Cafeteras",     799.0, "☕",
     "Molinillo integrado, LatteCrema System, pantalla táctil color"),
    ("Aspiradora Dyson V15 Detect Absolute",        "Hogar y Decoración", "Electrodomésticos", 749.0, "🌀",
     "Láser revela polvo fino, HEPA filtro, 60 minutos, pantalla LCD"),
    ("Purificador de Aire HEPA Levoit Core 600S",   "Hogar y Decoración", "Aire",          229.0, "💨",
     "CADR 410 m³/h, HEPA H13, filtro combinado, app WiFi"),
    ("Humidificador Ultrasónico Levoit LV600",      "Hogar y Decoración", "Aire",           79.0, "💧",
     "6L, cálido/frío, aromaterapia, display táctil, whisper-quiet"),
    ("Cuadro Decorativo Canvas Urban Arte 60×80",   "Hogar y Decoración", "Decoración",     59.0, "🖼️",
     "Impresión giclée, bastidor canvas de pino, listo para colgar"),
    ("Alfombra de Yute Natural Nanimarquina",       "Hogar y Decoración", "Decoración",    249.0, "🪡",
     "Fibra yute 100%, ecológica, 160×230 cm, antideslizante"),
    ("Planta Artificial Monstera Deliciosa 120cm",  "Hogar y Decoración", "Decoración",     89.0, "🌿",
     "UV-resistente, maceta decorativa incluida, se ve real"),
    ("Estantería Flotante IKEA LACK 3 piezas",      "Hogar y Decoración", "Muebles",        39.0, "📚",
     "Madera lacada, 110 kg carga, diferentes colores, fácil montaje"),
    ("Caja de Almacenamiento Kallax IKEA",          "Hogar y Decoración", "Muebles",       149.0, "📦",
     "4 estantes, madera lacada, 77×77 cm, compatible con inserciones"),
    ("Cortina Opaca Thermal Arlington",             "Hogar y Decoración", "Textiles",       45.0, "🪟",
     "Reduce ruido y frío, 2 paneles, 142×213 cm, varios colores"),
    ("Difusor de Aroma MUJI Ultrasónico",           "Hogar y Decoración", "Bienestar",      49.0, "🕯️",
     "60ml, difusión fría, LED ajustable, auto-apagado"),
    ("Maceta Cerámica Terracota Grande 30cm",       "Hogar y Decoración", "Decoración",     35.0, "🪴",
     "Cerámica artesanal, drenaje inferior, diámetro 30 cm"),
    ("Microondas Panasonic NN-CD87KS Inverter",     "Hogar y Decoración", "Electrodomésticos", 349.0,"📡",
     "1000W inverter, grill, convección, 32L, digital"),
    ("Armario Pax IKEA Blanco 200×236cm",           "Hogar y Decoración", "Muebles",       799.0, "🚪",
     "Marco blanco, puertas con espejo opcionales, accesorios internos"),
    ("Lavarropa Samsung WW22T4420AW 7kg",           "Hogar y Decoración", "Electrodomésticos", 599.0,"🫧",
     "7kg, A+++, eco bubble, 1400 rpm, smart control"),
    ("Heladera NoFrost LG GBB72PZDMN 384L",        "Hogar y Decoración", "Electrodomésticos", 899.0,"❄️",
     "Clase A++, Total No Frost, Door Cooling+, 384L"),
    ("Set Toallas Egyptian Cotton 6 piezas",        "Hogar y Decoración", "Baño",           59.0, "🛁",
     "Algodón egipcio 600 GSM, absorbente, 2 de baño + 2 manos + 2 cara"),
    ("Olla a Presión Instant Pot Duo 7-en-1 8L",    "Hogar y Decoración", "Cocina",         119.0, "🥘",
     "Cocina presión, slow, arroz, vaporera, saltear, app connect"),
    # Subtotal Hogar: 30 productos ──────────────────────────────────────────────

    # ── DEPORTES Y FITNESS (15% = 75 productos) ────────────────────────────────
    ("Mancuernas Ajustables PowerBlock 20kg",       "Deportes y Fitness", "Pesas",          129.0, "🏋️",
     "Sistema selector rápido, acero inox, reemplaza 9 pares"),
    ("Colchoneta Yoga Gaiam Premium Print 6mm",     "Deportes y Fitness", "Yoga",            39.0, "🧘",
     "PVC libre de látex, textura antideslizante, correa incluida"),
    ("Botella Térmica Hydro Flask 32oz",            "Deportes y Fitness", "Hidratación",     49.0, "💧",
     "Acero inox doble pared, TempShield 24h frío/12h calor"),
    ("Proteína Whey Gold Standard 5lb",             "Deportes y Fitness", "Nutrición",       89.0, "💪",
     "24g proteína/servicio, 5.5g BCAA, varios sabores"),
    ("Bicicleta Estática Peloton Bike+",            "Deportes y Fitness", "Cardio",        2495.0, "🚴",
     "Pantalla táctil 23,8\", instructor en vivo, inclinación automática"),
    ("Correa Fitness Garmin Fenix 7",               "Deportes y Fitness", "Wearables",      799.0, "⌚",
     "GPS + sats, maps integrados, 18 días batería, buceo 100m"),
    ("Soga de Saltar Speed Rope RX Jump",           "Deportes y Fitness", "Cardio",          29.0, "🤸",
     "Cable acero trenzado, mangos de aluminio, longitud regulable"),
    ("Pelota de Fútbol Adidas Al Rihla Pro League",  "Deportes y Fitness", "Fútbol",         39.0, "⚽",
     "FIFA Quality Pro, 100% poliuretano texturizado"),
    ("Banda Elástica Resistencia Set 5 TheraGun",   "Deportes y Fitness", "Resistencia",     29.0, "🏋️",
     "Loop bands en 5 niveles, látex natural, 30yd de longitud"),
    ("Raqueta de Tenis Wilson Blade 98 v8",         "Deportes y Fitness", "Tenis",          249.0, "🎾",
     "Carbon Frame, Countervail Technology, 305g, 16x19"),
    ("Guantes de Boxeo Everlast ProStyle 16oz",     "Deportes y Fitness", "Boxeo",           44.0, "🥊",
     "Foam triple densidad, cierre velcro, refuerzo pulgar"),
    ("Esterilla Pilates STOTT 190×55cm",            "Deportes y Fitness", "Pilates",         79.0, "🧘",
     "Espuma de alta densidad, antideslizante, bolsa incluida"),
    ("Masajeador Percusivo Theragun Pro Gen 5",     "Deportes y Fitness", "Recuperación",   499.0, "💆",
     "2400 rpm, 5 cabezales, app connect, 5h batería"),
    ("Tobilleras de Peso Liveup 2x2kg",             "Deportes y Fitness", "Accesorios",      24.0, "🏋️",
     "Ajustable, neopreno, velcro, 0.5-3 kg intercambiables"),
    ("Rodillo Foam Rolling Grid TriggerPoint",      "Deportes y Fitness", "Recuperación",    39.0, "🔄",
     "EVA multicapa, 3 superficies distintas, 33 cm"),
    ("Pantalón Hiking Fjallraven Keb Trousers",     "Deportes y Fitness", "Outdoor",        249.0, "🏔️",
     "G-1000 eco, zip-off 2en1, water-resistant, múltiples bolsillos"),
    ("Mochila Hidratación Camelback Octane 12L",    "Deportes y Fitness", "Hidratación",     99.0, "🎒",
     "Depósito 2.5L Crux, 10L carga, tiras ventiladas"),
    ("Botas de Trekking Salomon X Ultra 4 GTX",     "Deportes y Fitness", "Calzado Outdoor",160.0, "🥾",
     "Gore-Tex, Contagrip TA suela, senderismo exigente"),
    ("Casco Ciclismo Giro Syntax MIPS",             "Deportes y Fitness", "Ciclismo",       149.0, "🚲",
     "MIPS, 18 orificios, visera magnética, CE EN1078"),
    ("GPS Running Garmin Forerunner 265",           "Deportes y Fitness", "Running",        449.0, "⌚",
     "AMOLED, multiband GPS, métricas carrera avanzadas, 15d batería"),
    ("Piscina Hinchable Intex Ultra Frame 300×200", "Deportes y Fitness", "Natación",       249.0, "🏊",
     "Panel lateral robusto, filtro incluido, 4485L"),
    ("Tabla de Surf Foam Maluku 7ft",               "Deportes y Fitness", "Surf",           249.0, "🏄",
     "EPS + XPS, foam deck, fin tri-fin incluidas, principiantes"),
    ("Cuerda de Escalada Petzl Arial 9.5mm 60m",   "Deportes y Fitness", "Escalada",       229.0, "🧗",
     "Golden Dry, medio tratamiento, UIAA certificada"),
    ("Raqueta de Padel Head Graphene 360 Alpha Pro", "Deportes y Fitness", "Pádel",         199.0, "🏸",
     "Graphene 360, rombo con agujeros, intermedios-avanzados"),
    ("Kayak Inflable Advanced Elements AE1007",     "Deportes y Fitness", "Agua",           849.0, "🚣",
     "3 cámaras, armazón de aluminio plegable, 140 kg carga"),

    # ── LIBROS Y EDUCACIÓN (10% = 50 productos) ────────────────────────────────
    ("Curso Python para Data Science — Udemy",      "Libros y Educación", "Cursos Online",   49.0, "🐍",
     "120 horas, pandas/sklearn/tensorflow, certificado incluido"),
    ("'Atomic Habits' James Clear",                 "Libros y Educación", "Libros",          18.0, "📖",
     "Cómo construir buenos hábitos y romper los malos, bestseller global"),
    ("'Deep Learning' Goodfellow, Bengio & Courville","Libros y Educación","Libros Técnicos", 65.0, "📘",
     "La biblia del deep learning: RNN, CNN, GANs, fundamentos matemáticos"),
    ("Kindle Paperwhite 11th Gen",                  "Libros y Educación", "Lectores",       139.0, "📱",
     "6.8\", 300 ppi, luz ajustable, IPX8, 10 semanas batería"),
    ("'The Pragmatic Programmer' Hunt & Thomas",    "Libros y Educación", "Libros Técnicos",  49.0, "📗",
     "Edición 25 aniversario, DRY, Clean Code, arquitectura"),
    ("Suscripción Coursera Plus 1 año",             "Libros y Educación", "Cursos Online",  399.0, "🎓",
     "7000+ cursos, Google/IBM/Meta, certificados de Google Career"),
    ("Libro 'Python Crash Course' Eric Matthes",   "Libros y Educación", "Libros Técnicos",  35.0, "📙",
     "3ra edición, python 3.11+, proyectos: juego, datos, web"),
    ("'Designing Data-Intensive Applications' Kleppmann","Libros y Educación","Libros Técnicos",55.0,"📘",
     "Sistemas distribuidos, bases de datos, stream processing"),
    ("Calculadora Científica Casio FX-991EX",       "Libros y Educación", "Papelería",       24.0, "🔢",
     "552 funciones, visualización natural, calculadora vectorial"),
    ("'The Art of War' Sun Tzu Illus. Edition",     "Libros y Educación", "Libros",          22.0, "📖",
     "Edición ilustrada con caligrafía, comentarios históricos"),
    ("Suscripción Duolingo Super 1 año",            "Libros y Educación", "Idiomas",         79.0, "🦜",
     "Sin anuncios, historias, estadísticas, offline, 40 idiomas"),
    ("Libro 'Thinking, Fast and Slow' Kahneman",   "Libros y Educación", "Libros",          16.0, "📖",
     "Psicología de las decisiones, Nobel Economía 2002"),
    ("Diccionario RAE 23ava Edición",              "Libros y Educación", "Libros",          49.0, "📚",
     "93.111 entradas, tapa dura, versión oficial digital incluida"),
    ("'Clean Code' Robert C. Martin",              "Libros y Educación", "Libros Técnicos",  45.0, "📗",
     "Refactoring, SOLID, buenas prácticas, nombres significativos"),
    ("Curso AWS Solutions Architect — A Cloud Guru","Libros y Educación", "Cursos Online",   29.0, "☁️",
     "SAA-C03, laboratorios hands-on, simulacros de examen"),
    ("Libro 'The Lean Startup' Eric Ries",         "Libros y Educación", "Libros",          17.0, "📖",
     "MVP, Build-Measure-Learn, iteración rápida, pivot"),
    ("Tablet Amazon Fire 7 Kids Edition",          "Libros y Educación", "Lectores",         99.0, "📱",
     "Android for Kids, 6000+ apps/libros/videos, funda incluida"),
    ("'The Midnight Library' Matt Haig",           "Libros y Educación", "Libros",          14.0, "📖",
     "Bestseller NYT, aventura emocional, 2022 Goodreads Choice"),
    ("'Statistics' Freedman Pisani Purves",        "Libros y Educación", "Libros Técnicos",  89.0, "📘",
     "4ta edición, inferencia, regresión, tablas de contingencia"),
    ("Curso Machine Learning — Stanford (Coursera)","Libros y Educación","Cursos Online",    49.0, "🤖",
     "Andrew Ng, 56h, regresión/clasificación/redes neuronales"),
    ("Libro 'Shoe Dog' Phil Knight",               "Libros y Educación", "Libros",          16.0, "📖",
     "Memorias del fundador de Nike, historia emprendedora"),
    # Subtotal Libros: 20 productos ─────────────────────────────────────────────

    # ── BELLEZA Y CUIDADO PERSONAL (10% = 50 productos) ────────────────────────
    ("Perfume Carolina Herrera Good Girl EDP 80ml", "Belleza y Cuidado Personal", "Perfumes", 109.0, "🌹",
     "Notas de jazmín y cacao, frasco original tacón de mujer"),
    ("Crema Hidratante Neutrogena Hydro Boost 50ml","Belleza y Cuidado Personal", "Cremas",   22.0, "💧",
     "Ácido hialurónico, textura gel-agua, no comedogénica"),
    ("Secador Dyson Supersonic HD15",               "Belleza y Cuidado Personal", "Cabello", 429.0, "💇",
     "Motor Dyson V9, temperatura inteligente, 3 cepillos incluidos"),
    ("Plancha Ghd Platinum+ Styler",                "Belleza y Cuidado Personal", "Cabello", 249.0, "💇",
     "Tecnología predictiva, 185°C, bisagra flexible, cable giratorio"),
    ("Afeitadora Philips Norelco 9500",             "Belleza y Cuidado Personal", "Afeitado", 149.0, "🪒",
     "V-Track Precision Blades, 100% seco/húmedo, app connect"),
    ("Cepillo Eléctrico Oral-B IO Series 9",        "Belleza y Cuidado Personal", "Higiene",  199.0, "🦷",
     "IA reconoce zona, presión inteligente, 6 modos, display OLED"),
    ("Paleta Eyeshadow Urban Decay Naked 3",        "Belleza y Cuidado Personal", "Maquillaje", 54.0, "👁️",
     "12 sombras rosas-cobre, acabado matte/shimmer/metallic"),
    ("Base Maquillaje Fenty Beauty Pro Filt'r 30ml","Belleza y Cuidado Personal", "Maquillaje", 39.0, "💄",
     "50 tonos, larga duración 24h, acabado semi-matte, Rihanna"),
    ("Sérum Vitamina C The Ordinary 10%",           "Belleza y Cuidado Personal", "Cremas",   11.0, "🌟",
     "AA 2G, ácido ascórbico estable, antioxidante, iluminador"),
    ("Perfume Bleu de Chanel EDT 100ml",            "Belleza y Cuidado Personal", "Perfumes", 119.0, "🔵",
     "Madera aromática, frescor citrus, masculino atemporal"),
    ("Máscara de Pestañas NARS Climax",             "Belleza y Cuidado Personal", "Maquillaje", 26.0, "👁️",
     "Volumen extremo, fórmula buildable, no transfiere"),
    ("Tónico Facial Paula's Choice BHA 2%",         "Belleza y Cuidado Personal", "Cremas",   32.0, "🧴",
     "Ácido salicílico 2%, poros dilatados, textura refinada"),
    ("Bálsamo Labial Carmex Classic Pot",           "Belleza y Cuidado Personal", "Labios",    4.0, "💋",
     "Mentol + alcanfor, ultra hidratante, SPF 15"),
    ("Set Pinzas Tweezerman 4 piezas",              "Belleza y Cuidado Personal", "Cuidado",  29.0, "✂️",
     "Acero inox, afiladas de precisión, estuche incluido"),
    ("Aceite Capilar Moroccanoil Original 100ml",   "Belleza y Cuidado Personal", "Cabello",  44.0, "💧",
     "Argan oil, keratin, brillo y suavidad inmediatos"),
    ("Gel Ducha Aesop Geranium Leaf Body Cleanser", "Belleza y Cuidado Personal", "Cuerpo",  39.0, "🛁",
     "Geranio + Ylang Ylang, pH balanceado, sin sulfatos"),
    ("Protector Solar Supergoop Unseen SPF 40",     "Belleza y Cuidado Personal", "Protección Solar", 34.0, "☀️",
     "Oil-free, invisible, no whitening, base perfecta antes del maquillaje"),
    ("Rizadora Dyson Airwrap Complete Long",        "Belleza y Cuidado Personal", "Cabello",  599.0, "💇",
     "Air-wrapping sin calor extremo, 6 accesorios, 3 temperaturas"),
    ("Rubor NARS Orgasm El Original",               "Belleza y Cuidado Personal", "Maquillaje", 30.0, "🌸",
     "Coral-melocotón con shimmer dorado, el más vendido de NARS"),
    ("Colonia Dolce & Gabbana Light Blue EDT 125ml","Belleza y Cuidado Personal", "Perfumes", 89.0, "💙",
     "Notas manzana siciliana, cedro, ámbar, mediterráneo y fresco"),
    # Subtotal Belleza: 20 productos ───────────────────────────────────────────

    # ── ALIMENTOS Y BEBIDAS (5% = 25 productos) ────────────────────────────────
    ("Café de Especialidad Colombia Huila 500g",    "Alimentos y Bebidas", "Café",           24.0, "☕",
     "Proceso washed, notas cacao y frutos rojos, tueste medio"),
    ("Té Verde Orgánico Clipper 40 bolsitas",       "Alimentos y Bebidas", "Infusiones",     12.0, "🍵",
     "100% orgánico, sin blanquear, Fairtrade, aroma fresco"),
    ("Proteína Vegana Garden of Life Sport 1.4lb",  "Alimentos y Bebidas", "Suplementos",    59.0, "🌱",
     "30g proteína, probióticos, B12, sin gluten, NSF certificado"),
    ("Aceite de Oliva Extra Virgen Marqués de Riscal 750ml","Alimentos y Bebidas","Aceites",  22.0, "🫒",
     "Primera presión frío, acidity <0.3%, DOP España"),
    ("Mix de Frutos Secos Premium Wonderful 1kg",   "Alimentos y Bebidas", "Snacks",         34.0, "🥜",
     "Almendras, nueces, anacardos, sin sal, sin aceite añadido"),
    ("Cápsulas Nespresso Original Intenso x50",     "Alimentos y Bebidas", "Café",           29.0, "☕",
     "Compatible Nespresso Original, intensidad 8, tueste oscuro"),
    ("Whisky Johnnie Walker Black Label 750ml",      "Alimentos y Bebidas", "Bebidas Alcohólicas", 39.0, "🥃",
     "12 años blended scotch, familia de 40% ABV, notas ahumadas"),
    ("Granola Artesanal Organic Kitchen 500g",       "Alimentos y Bebidas", "Cereales",       18.0, "🥣",
     "Avena, semillas y frutos rojos, sin azúcar refinada, horneado"),
    ("Chocolate Valrhona Caraïbe Dark 70% 250g",    "Alimentos y Bebidas", "Dulces",         29.0, "🍫",
     "Couverture de repostería, ganache, temperatura de trabajo 29-30°C"),
    ("Kombuch GT's Synergy Gingerberry 16oz",       "Alimentos y Bebidas", "Bebidas",         6.0, "🍶",
     "Raw organic, 1B probióticos, SCOBY cultivado, baja fermentación"),
    ("Canasta Picnic Gourmet Premium 20 productos", "Alimentos y Bebidas", "Regalos",        129.0, "🧺",
     "Queso, embutido, mermelada, chips, vino, entregado en canasta"),
    ("Vitaminas C 1000mg Nature Made 300 cápsulas", "Alimentos y Bebidas", "Suplementos",   19.0, "💊",
     "USP verified, 250mg bioflavonoides, sin gluten, liberación prolongada"),
]

random.seed(RANDOM_SEED)


def _build_cat_tree_map(ct: pd.DataFrame) -> dict:
    """
    Construye un mapa item_id → categoryid raíz.
    Recorre el árbol hacia arriba hasta encontrar un nodo sin parentid.
    """
    parent = {}
    for _, row in ct.iterrows():
        cid = int(row["categoryid"])
        pid = row["parentid"]
        parent[cid] = int(pid) if pd.notna(pid) else None

    def get_root(cid: int, depth: int = 0) -> int:
        if depth > 50:  # protección contra ciclos
            return cid
        p = parent.get(cid)
        if p is None:
            return cid
        return get_root(p, depth + 1)

    root_map = {}
    for cid in parent:
        root_map[cid] = get_root(cid)
    return root_map


def _load_item_categories(ip1_path: Path, ip2_path: Path) -> dict:
    """
    Lee item_properties y retorna {item_id: categoryid}.
    Solo lee la propiedad 'categoryid'.
    """
    item_cat = {}
    for path in [ip1_path, ip2_path]:
        if not path.exists():
            continue
        # Leer en chunks para ahorrar memoria
        for chunk in pd.read_csv(path, chunksize=200_000, dtype=str):
            mask = chunk["property"] == "categoryid"
            for _, row in chunk[mask].iterrows():
                iid = int(row["itemid"])
                try:
                    cid = int(row["value"].strip())
                    item_cat[iid] = cid
                except (ValueError, AttributeError):
                    pass
    return item_cat


def build_catalog():
    """Punto de entrada principal. Construye y guarda el catálogo."""
    print("=" * 60)
    print("build_product_catalog.py — Nexus Data Co.")
    print("=" * 60)

    # ── 1. Calcular top 500 ítems por frecuencia ───────────────────────────────
    print("\n[1/4] Cargando interaction_matrix.csv ...")
    im_path = DATA_PROC / "interaction_matrix.csv"
    if not im_path.exists():
        raise FileNotFoundError(f"No encontrado: {im_path}")

    im = pd.read_csv(im_path, usecols=["itemid"], dtype={"itemid": int})
    freq = im["itemid"].value_counts()
    top500_ids = list(freq.index[:TOP_N])
    total_items = im["itemid"].nunique()
    print(f"    Total ítems únicos: {total_items:,}")
    print(f"    Top 500 (umbral mínimo): {freq.iloc[TOP_N-1]} interacciones")

    # ── 2. Cargar árbol de categorías ─────────────────────────────────────────
    print("\n[2/4] Cargando category_tree.csv ...")
    ct_path = DATA_RAW / "category_tree.csv"
    if ct_path.exists():
        ct = pd.read_csv(ct_path)
        root_map = _build_cat_tree_map(ct)
    else:
        root_map = {}
        print("    WARNING: category_tree.csv no encontrado, usando categoría 'Otros'")

    # ── 3. Cargar categorías de ítems ─────────────────────────────────────────
    print("\n[3/4] Cargando item_properties (puede tardar 2-3 min) ...")
    ip1 = DATA_RAW / "item_properties_part1.csv"
    ip2 = DATA_RAW / "item_properties_part2.csv"
    item_cat_raw = _load_item_categories(ip1, ip2)
    print(f"    Ítems con categoría encontrada: {len(item_cat_raw):,}")

    # Resolver categoría raíz para cada ítem
    item_root_cat = {}
    for iid, cid in item_cat_raw.items():
        root_cid = root_map.get(cid, cid)
        cat_name = ROOT_CAT_NAMES.get(root_cid, "Otros")
        item_root_cat[iid] = cat_name

    # ── 4. Construir catálogo ─────────────────────────────────────────────────
    print("\n[4/4] Construyendo catálogo ...")

    items_dict = {}

    # Opción A: Top 500 ítems con productos reales
    productos_shuffled = OPCION_A_PRODUCTOS.copy()
    random.shuffle(productos_shuffled)
    # Asegurar que hay suficientes (rellenar si faltan)
    while len(productos_shuffled) < TOP_N:
        extra = productos_shuffled[:]
        random.shuffle(extra)
        productos_shuffled.extend(extra)

    for rank, item_id in enumerate(top500_ids):
        prod = productos_shuffled[rank % len(productos_shuffled)]
        nombre, cat, subcat, precio, emoji, desc = prod
        items_dict[str(item_id)] = {
            "item_id":     item_id,
            "name":        nombre,
            "category":    cat,
            "subcategory": subcat,
            "price":       precio,
            "currency":    "USD",
            "emoji":       emoji,
            "option":      "A",
            "description": desc,
            "rank_popularity": rank + 1,
        }

    # Opción B: Resto del catálogo
    all_item_ids = freq.index.tolist()  # todos los ítems del dataset
    top500_set = set(top500_ids)
    option_b_ids = [iid for iid in all_item_ids if iid not in top500_set]

    for item_id in option_b_ids:
        cat_name = item_root_cat.get(item_id, "Otros")
        emoji = CATEGORY_EMOJIS.get(cat_name, CATEGORY_EMOJIS["default"])
        items_dict[str(item_id)] = {
            "item_id":     item_id,
            "name":        f"{cat_name} — Ref. #{item_id}",
            "category":    cat_name,
            "subcategory": "General",
            "price":       None,
            "currency":    "USD",
            "emoji":       emoji,
            "option":      "B",
            "description": f"Producto del catálogo · ID verificado",
            "rank_popularity": None,
        }

    # Output final
    catalog = {
        "metadata": {
            "total_items":     len(items_dict),
            "option_a_count":  TOP_N,
            "option_b_count":  len(items_dict) - TOP_N,
            "generated_at":    str(date.today()),
            "description":     (
                f"Top {TOP_N} ítems con productos reales de e-commerce (Opción A), "
                "resto enriquecido con categoría del dataset (Opción B)"
            ),
        },
        "items": items_dict,
    }

    DATA_PROC.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Catálogo guardado en: {OUTPUT_PATH}")
    print(f"  Total ítems:  {len(items_dict):,}")
    print(f"  Opción A:     {TOP_N}")
    print(f"  Opción B:     {len(items_dict) - TOP_N:,}")
    print(f"  Generado el:  {date.today()}")
    print("=" * 60)

    # Mostrar primeras 5 entradas
    print("\nPrimeras 5 entradas del catálogo:")
    for i, (k, v) in enumerate(list(items_dict.items())[:5]):
        print(f"  [{k}]: {v['emoji']} {v['name']} — {v['category']} — ${v['price']}")


if __name__ == "__main__":
    build_catalog()
