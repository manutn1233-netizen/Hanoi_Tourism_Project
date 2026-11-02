import heapq
import time
from typing import Dict, List, Tuple, Optional

GRAPH: Dict[str, Dict[str, int]] = {
    'S': {'A': 16, 'B': 25, 'C': 19},
    'A': {'D': 15, 'E': 10, 'F': 7},
    'B': {'D': 7,  'E': 14, 'F': 13},
    'C': {'D': 11, 'E': 13, 'F': 6},
    'D': {'I': 17, 'J': 11, 'K': 10},
    'E': {'I': 10, 'J': 6,  'K': 5},
    'F': {'I': 8,  'J': 5,  'K': 14},
    'I': {'G': 10},
    'J': {'G': 12},
    'K': {'G': 13},
    'G': {}
}

HEURISTIC: Dict[str, int] = {
    'S': 240,
    'A': 180, 'B': 180, 'C': 180,
    'D': 105, 'E': 105, 'F': 105,
    'I': 30,  'J': 30,  'K': 30,
    'G': 0
}

LOCATION_NAMES: Dict[str, str] = {
    'S': 'Times City (Khởi hành)',
    'A': 'CV Thống Nhất (Chụp ảnh)',
    'B': 'Phố Phan Đình Phùng (Chụp ảnh)',
    'C': 'Hồ Gươm (Chụp ảnh)',
    'D': "Pizza 4P's Liễu Giai (Ăn uống)",
    'E': 'Kkul BBQ Nam Ngư (Ăn uống)',
    'F': 'Haidilao Phạm Ngọc Thạch (Ăn uống)',
    'I': '9981 Coffee (Cà phê)',
    'J': 'Horizon Cafe (Cà phê)',
    'K': 'Hilltop Cafe (Cà phê)',
    'G': 'Lăng Bác (Đích đến)'
}


def a_star_search(start: str, goal: str) -> Tuple[Optional[List[str]], int, int]:
    """
    Thuật toán A* Search: Đánh giá dựa trên f(u) = g(u) + h(u)
    Trả về: (Lộ trình, Tổng chi phí di chuyển, Số bước duyệt)
    """
    
    open_set: List[Tuple[int, int, str, List[str]]] = []
    heapq.heappush(open_set, (HEURISTIC[start], 0, start, [start]))
    
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'📌 BẢNG TRACE THỰC THI THUẬT TOÁN A* SEARCH':<76} │")
    print("├" + "─" * 6 + "┬" + "─" * 10 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 8 + "┬" + "─" * 32 + "┤")
    print(f"│ {'Bước':<4} │ {'Nút xét':<8} │ {'g(u)':<6} │ {'h(u)':<6} │ {'f(u)':<6} │ {'Lộ trình hiện tại':<30} │")
    print("├" + "─" * 6 + "┼" + "─" * 10 + "┼" + "─" * 8 + "┼" + "─" * 8 + "┼" + "─" * 8 + "┼" + "─" * 32 + "┤")
    
    step = 0
    while open_set:
        f, g, curr, path = heapq.heappop(open_set)
        path_str = " ➔ ".join(path)
        print(f"│ Step {step:<2} │ {curr:<8} │ {g:<4}m  │ {HEURISTIC[curr]:<6} │ {f:<6} │ {path_str:<30} │")
        
        if curr == goal:
            print("└" + "─" * 6 + "┴" + "─" * 10 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 32 + "┘")
            return path, g, step
            
        for neighbor, weight in GRAPH[curr].items():
            g_new = g + weight
            f_new = g_new + HEURISTIC[neighbor]
            heapq.heappush(open_set, (f_new, g_new, neighbor, path + [neighbor]))
        step += 1
        
    print("└" + "─" * 6 + "┴" + "─" * 10 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 8 + "┴" + "─" * 32 + "┘")
    return None, float('inf'), step


def gbfs_search(start: str, goal: str) -> Tuple[Optional[List[str]], int, int]:
    """
    Thuật toán Greedy Best-First Search: Đánh giá dựa trên f(u) = h(u)
    Trả về: (Lộ trình, Tổng chi phí di chuyển, Số bước duyệt)
    """
  
    open_set: List[Tuple[int, str, int, List[str]]] = []
    heapq.heappush(open_set, (HEURISTIC[start], start, 0, [start]))
    
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'📌 BẢNG TRACE THỰC THI THUẬT TOÁN GREEDY (GBFS)':<76} │")
    print("├" + "─" * 6 + "┬" + "─" * 10 + "┬" + "─" * 8 + "┬" + "─" * 12 + "┬" + "─" * 36 + "┤")
    print(f"│ {'Bước':<4} │ {'Nút xét':<8} │ {'h(u)':<6} │ {'g tích lũy':<10} │ {'Lộ trình hiện tại':<34} │")
    print("├" + "─" * 6 + "┼" + "─" * 10 + "┼" + "─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 36 + "┤")
    
    step = 0
    while open_set:
        h_val, curr, g, path = heapq.heappop(open_set)
        path_str = " ➔ ".join(path)
        print(f"│ Step {step:<2} │ {curr:<8} │ {h_val:<6} │ {g:<8}m   │ {path_str:<34} │")
        
        if curr == goal:
            print("└" + "─" * 6 + "┴" + "─" * 10 + "┴" + "─" * 8 + "┴" + "─" * 12 + "┴" + "─" * 36 + "┘")
            return path, g, step
            
        for neighbor, weight in GRAPH[curr].items():
            g_new = g + weight
            heapq.heappush(open_set, (HEURISTIC[neighbor], neighbor, g_new, path + [neighbor]))
        step += 1
        
    print("└" + "─" * 6 + "┴" + "─" * 10 + "┴" + "─" * 8 + "┴" + "─" * 12 + "┴" + "─" * 36 + "┘")
    return None, float('inf'), step


def print_itinerary(path: List[str]) -> None:
    """Hiển thị thời gian biểu thực tế chi tiết từng chặng cho Lớp trưởng."""
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'🗓️  LỊCH TRÌNH CHI TIẾT TRONG NGÀY CHO LỐP TRƯỞNG':<73} │")
    print("├" + "─" * 18 + "┬" + "─" * 43 + "┬" + "─" * 15 + "┤")
    print(f"│ {'Khung giờ':<16} │ {'Hoạt động & Địa điểm':<41} │ {'Thời gian':<13} │")
    print("├" + "─" * 18 + "┼" + "─" * 43 + "┼" + "─" * 15 + "┤")
    

    durations = {'Photo': 45, 'Food': 75, 'Coffee': 45}
    current_time_min = 8 * 60  
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        travel_time = GRAPH[u][v]
        
        start_str = f"{current_time_min // 60:02d}:{current_time_min % 60:02d}"
        current_time_min += travel_time
        end_str = f"{current_time_min // 60:02d}:{current_time_min % 60:02d}"
        move_desc = f"🛵 Xe máy: {u} ➔ {v} ({LOCATION_NAMES[v]})"
        print(f"│ {start_str} - {end_str:<7} │ {move_desc:<41} │ Đi xe {travel_time:<2} phút │")
        
        if v != 'G':
            activity_type = 'Photo' if v in ['A', 'B', 'C'] else ('Food' if v in ['D', 'E', 'F'] else 'Coffee')
            stop_time = durations[activity_type]
            
            stop_start_str = f"{current_time_min // 60:02d}:{current_time_min % 60:02d}"
            current_time_min += stop_time
            stop_end_str = f"{current_time_min // 60:02d}:{current_time_min % 60:02d}"
            
            act_names = {'Photo': '📸 Chụp ảnh check-in', 'Food': '🍽️  Ăn trưa trải nghiệm', 'Coffee': '☕ Cà phê nghỉ ngơi'}
            stop_desc = f"{act_names[activity_type]} tại {v}"
            print(f"│ {stop_start_str} - {stop_end_str:<7} │ {stop_desc:<41} │ Dừng {stop_time:<2} phút │")

    print("└" + "─" * 18 + "┴" + "─" * 43 + "┴" + "─" * 15 + "┘")


def print_comparison_table(cost_a: int, steps_a: int, cost_g: int, steps_g: int) -> None:
    """In bảng so sánh đối chiếu hiệu năng giữa A* và GBFS."""
    diff = cost_g - cost_a
    percent = (diff / cost_g) * 100 if cost_g > 0 else 0
    
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'📊 BẢNG SO SÁNH HIỆU NĂNG THUẬT TOÁN (A* vs GBFS)':<73} │")
    print("├" + "─" * 28 + "┬" + "─" * 24 + "┬" + "─" * 24 + "┤")
    print(f"│ {'Tiêu chí so sánh':<26} │ {'A* Search Algorithm':<22} │ {'Greedy Search (GBFS)':<22} │")
    print("├" + "─" * 28 + "┼" + "─" * 24 + "┼" + "─" * 24 + "┤")
    print(f"│ {'Tổng thời gian di chuyển':<26} │ {str(cost_a) + ' phút (Tối ưu 100%)':<22} │ {str(cost_g) + ' phút':<22} │")
    print(f"│ {'Số bước mở rộng (Steps)':<26} │ {str(steps_a) + ' bước':<22} │ {str(steps_g) + ' bước':<22} │")
    print(f"│ {'Tính tối ưu toàn cục':<26} │ {'✅ Cam kết Tối ưu':<22} │ {'❌ Bị sa lầy cục bộ':<22} │")
    print(f"│ {'Chênh lệch hiệu năng':<26} │ {f'⚡ Tiết kiệm {diff}p ({percent:.1f}%)':<22} │ {f'🐢 Chậm hơn {diff} phút':<22} │")
    print("└" + "─" * 28 + "┴" + "─" * 24 + "┴" + "─" * 24 + "┘\n")

if __name__ == "__main__":
    start_node, goal_node = 'S', 'G'

    print("\n" + "═" * 80)
    print(" 🚀 HỆ THỐNG TƯ VẤN LỊCH TRÌNH DU LỊCH HÀ NỘI BẰNG AI - LỐP TRƯỜNG TRIP PLANNER ")
    print("═" * 80)

    path_a, cost_a, steps_a = a_star_search(start_node, goal_node)

    path_g, cost_g, steps_g = gbfs_search(start_node, goal_node)

    print_comparison_table(cost_a, steps_a, cost_g, steps_g)

    if path_a:
        print_itinerary(path_a)
