"""
クリスマスツリー2次元パッキング問題 - 改善版
Bottom-Left戦略の正しい実装（左下への詰め込み）
"""

import math
import random
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from shapely import affinity
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm import tqdm

# 精度設定
getcontext().prec = 25
SCALE_FACTOR = Decimal('1e15')


@dataclass
class TreeParameters:
    """クリスマスツリーの形状パラメータ"""
    trunk_w: Decimal = Decimal('0.15')
    trunk_h: Decimal = Decimal('0.2')
    base_w: Decimal = Decimal('0.7')
    mid_w: Decimal = Decimal('0.4')
    top_w: Decimal = Decimal('0.25')
    tip_y: Decimal = Decimal('0.8')
    tier_1_y: Decimal = Decimal('0.5')
    tier_2_y: Decimal = Decimal('0.25')
    base_y: Decimal = Decimal('0.0')
    
    @property
    def trunk_bottom_y(self):
        return -self.trunk_h


class ChristmasTree:
    """クリスマスツリー（Shapely版、高精度）"""
    
    def __init__(self, center_x='0', center_y='0', angle='0', params=None, use_scale=True):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        self.params = params or TreeParameters()
        self.use_scale = use_scale
        
        self.polygon = self._create_polygon()
        self.prepared_polygon = prep(self.polygon)
    
    def _create_polygon(self) -> ShapelyPolygon:
        """ツリーのポリゴンを生成"""
        p = self.params
        
        vertices = [
            (Decimal('0.0'), p.tip_y),
            (p.top_w / 2, p.tier_1_y),
            (p.top_w / 4, p.tier_1_y),
            (p.mid_w / 2, p.tier_2_y),
            (p.mid_w / 4, p.tier_2_y),
            (p.base_w / 2, p.base_y),
            (p.trunk_w / 2, p.base_y),
            (p.trunk_w / 2, p.trunk_bottom_y),
            (-p.trunk_w / 2, p.trunk_bottom_y),
            (-p.trunk_w / 2, p.base_y),
            (-p.base_w / 2, p.base_y),
            (-p.mid_w / 4, p.tier_2_y),
            (-p.mid_w / 2, p.tier_2_y),
            (-p.top_w / 4, p.tier_1_y),
            (-p.top_w / 2, p.tier_1_y),
        ]
        
        if self.use_scale:
            scaled_vertices = [
                (float(x * SCALE_FACTOR), float(y * SCALE_FACTOR))
                for x, y in vertices
            ]
        else:
            scaled_vertices = [(float(x), float(y)) for x, y in vertices]
        
        initial_polygon = ShapelyPolygon(scaled_vertices)
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        
        if self.use_scale:
            translated = affinity.translate(
                rotated,
                xoff=float(self.center_x * SCALE_FACTOR),
                yoff=float(self.center_y * SCALE_FACTOR)
            )
        else:
            translated = affinity.translate(
                rotated,
                xoff=float(self.center_x),
                yoff=float(self.center_y)
            )
        
        return translated
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """境界を取得 (minx, miny, maxx, maxy)"""
        return self.polygon.bounds
    
    def overlaps(self, other: 'ChristmasTree', tolerance: float = 1e-6) -> bool:
        """他のツリーと重なっているか（接触は許可）"""
        if not self.prepared_polygon.intersects(other.polygon):
            return False
        
        intersection = self.polygon.intersection(other.polygon)
        
        if intersection.is_empty or intersection.area < tolerance:
            return False
        
        return True
    
    def distance_to(self, other: 'ChristmasTree') -> float:
        """他のツリーとの距離"""
        return self.polygon.distance(other.polygon)


class ImprovedBottomLeftPacker:
    """
    改善版Bottom-Left Packer
    正しいBottom-Left戦略: 左下への詰め込みを優先
    """
    
    def __init__(
        self,
        grid_step: float = 0.02,  # グリッドステップを少し粗くして高速化
        use_scale: bool = True,
        max_search_width: float = 20.0,  # 探索範囲の上限
        max_search_height: float = 20.0
    ):
        self.grid_step = grid_step
        self.use_scale = use_scale
        self.max_search_width = max_search_width
        self.max_search_height = max_search_height
    
    def pack(
        self,
        sequence: List[int],
        angles: List[int],
        verbose: bool = False
    ) -> Tuple[List[ChristmasTree], float]:
        """
        ツリーを配置
        
        Args:
            sequence: 配置順序
            angles: 各ツリーの回転角度
            verbose: 詳細出力
            
        Returns:
            placed_trees: 配置されたツリー
            container_size: 正方形コンテナのサイズ
        """
        placed_trees = []
        
        iterator = sequence if not verbose else tqdm(sequence, desc="Packing trees")
        
        for idx in iterator:
            angle = angles[idx]
            
            if not placed_trees:
                # 最初のツリーは左下隅(0,0)に配置
                tree = ChristmasTree(
                    center_x='0',
                    center_y='0',
                    angle=str(angle),
                    use_scale=self.use_scale
                )
                # 境界を調整して左下隅に配置
                minx, miny, _, _ = tree.get_bounds()
                if self.use_scale:
                    tree = ChristmasTree(
                        center_x=str(Decimal(str(-minx / float(SCALE_FACTOR)))),
                        center_y=str(Decimal(str(-miny / float(SCALE_FACTOR)))),
                        angle=str(angle),
                        use_scale=self.use_scale
                    )
                else:
                    tree = ChristmasTree(
                        center_x=str(Decimal(str(-minx))),
                        center_y=str(Decimal(str(-miny))),
                        angle=str(angle),
                        use_scale=self.use_scale
                    )
                placed_trees.append(tree)
            else:
                # Bottom-Left戦略で配置位置を探索
                best_position = self._find_bottom_left_position(
                    placed_trees,
                    angle,
                    verbose=False
                )
                
                tree = ChristmasTree(
                    center_x=str(best_position[0]),
                    center_y=str(best_position[1]),
                    angle=str(angle),
                    use_scale=self.use_scale
                )
                placed_trees.append(tree)
        
        # 正方形コンテナサイズを計算
        container_size = self._calculate_square_container_size(placed_trees)
        
        return placed_trees, container_size
    
    def _find_bottom_left_position(
        self,
        placed_trees: List[ChristmasTree],
        angle: int,
        verbose: bool = False
    ) -> Tuple[Decimal, Decimal]:
        """
        真のBottom-Left戦略: 左下からの距離を最小化
        
        評価関数: sqrt(x^2 + y^2) を最小化
        つまり、原点(0,0)からの距離を最小化
        """
        # 現在の配置範囲を取得
        all_bounds = [tree.get_bounds() for tree in placed_trees]
        if self.use_scale:
            max_x = max(b[2] for b in all_bounds) / float(SCALE_FACTOR)
            max_y = max(b[3] for b in all_bounds) / float(SCALE_FACTOR)
        else:
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
        
        # 探索範囲を設定
        search_margin = 2.0
        x_max = min(max_x + search_margin, self.max_search_width)
        y_max = min(max_y + search_margin, self.max_search_height)
        
        best_pos = None
        best_score = float('inf')
        
        # Bottom-Left探索: 左下からの距離を評価
        # より効率的な探索順序: 左下から右上へ
        y = 0.0
        while y <= y_max:
            x = 0.0
            while x <= x_max:
                # 候補位置にツリーを配置
                candidate = ChristmasTree(
                    center_x=str(Decimal(str(x))),
                    center_y=str(Decimal(str(y))),
                    angle=str(angle),
                    use_scale=self.use_scale
                )
                
                # 重なりチェック
                has_overlap = False
                for tree in placed_trees:
                    if candidate.overlaps(tree):
                        has_overlap = True
                        break
                
                if not has_overlap:
                    # 左下からの距離を計算（ユークリッド距離）
                    # より左下に配置されるほどスコアが小さくなる
                    distance = math.sqrt(x**2 + y**2)
                    
                    # ボーナス: yを優先的に小さくする（わずかな重み付け）
                    score = distance + y * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (Decimal(str(x)), Decimal(str(y)))
                        
                        if verbose:
                            print(f"  Better position: ({x:.3f}, {y:.3f}), score: {score:.3f}")
                
                x += self.grid_step
            
            y += self.grid_step
        
        if best_pos is None:
            # 見つからない場合は右上に配置
            print(f"Warning: No valid position found, placing at ({x_max + 1:.3f}, {y_max + 1:.3f})")
            best_pos = (Decimal(str(x_max + 1.0)), Decimal(str(y_max + 1.0)))
        
        return best_pos
    
    def _calculate_square_container_size(self, trees: List[ChristmasTree]) -> float:
        """正方形コンテナのサイズを計算（max(width, height)）"""
        if not trees:
            return 0.0
        
        all_bounds = [tree.get_bounds() for tree in trees]
        
        if self.use_scale:
            min_x = min(b[0] for b in all_bounds) / float(SCALE_FACTOR)
            min_y = min(b[1] for b in all_bounds) / float(SCALE_FACTOR)
            max_x = max(b[2] for b in all_bounds) / float(SCALE_FACTOR)
            max_y = max(b[3] for b in all_bounds) / float(SCALE_FACTOR)
        else:
            min_x = min(b[0] for b in all_bounds)
            min_y = min(b[1] for b in all_bounds)
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 正方形制約: max(width, height)
        return max(width, height)


class Individual:
    """GA個体"""
    
    def __init__(self, n_trees: int, angle_options: List[int]):
        self.n_trees = n_trees
        self.angle_options = angle_options
        self.sequence = list(range(n_trees))
        random.shuffle(self.sequence)
        self.angles = [random.choice(angle_options) for _ in range(n_trees)]
        self.fitness = None
        self.container_size = None
        self.trees = None
    
    def copy(self) -> 'Individual':
        ind = Individual(self.n_trees, self.angle_options)
        ind.sequence = self.sequence.copy()
        ind.angles = self.angles.copy()
        ind.fitness = self.fitness
        ind.container_size = self.container_size
        ind.trees = self.trees.copy() if self.trees else None
        return ind


class ImprovedGeneticAlgorithm:
    """改善版遺伝的アルゴリズム"""
    
    def __init__(
        self,
        n_trees: int,
        angle_options: List[int],
        population_size: int = 50,
        elite_ratio: float = 0.1,
        mutation_rate: float = 0.15,
        max_generations: int = 40,
        grid_step: float = 0.02,
        impact_factor: float = 1000.0
    ):
        self.n_trees = n_trees
        self.angle_options = angle_options
        self.population_size = population_size
        self.elite_size = int(population_size * elite_ratio)
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.impact_factor = impact_factor
        
        self.packer = ImprovedBottomLeftPacker(grid_step=grid_step, use_scale=True)
        self.population = []
        self.best_individual = None
        self.fitness_history = []
    
    def initialize_population(self):
        """初期集団を生成"""
        self.population = [
            Individual(self.n_trees, self.angle_options)
            for _ in range(self.population_size)
        ]
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """適応度を計算"""
        trees, container_size = self.packer.pack(
            individual.sequence,
            individual.angles,
            verbose=False
        )
        
        individual.trees = trees
        individual.container_size = container_size
        
        # 適応度 = impact_factor / container_size^2
        # 正方形の面積を考慮
        fitness = self.impact_factor / (container_size ** 2 + 1e-6)
        
        return fitness
    
    def evaluate_population(self):
        """集団全体の適応度を評価"""
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self.evaluate_fitness(ind)
    
    def selection(self) -> List[Individual]:
        """選択（エリート + ルーレット）"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.elite_size]
        
        fitness_sum = sum(ind.fitness for ind in self.population)
        if fitness_sum == 0:
            selected = random.choices(self.population, k=self.population_size - self.elite_size)
        else:
            probs = [ind.fitness / fitness_sum for ind in self.population]
            selected = random.choices(
                self.population,
                weights=probs,
                k=self.population_size - self.elite_size
            )
        
        return elites + selected
    
    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """交叉（順序交叉）"""
        size = self.n_trees

        o1 = Individual(self.n_trees, self.angle_options)
        o2 = Individual(self.n_trees, self.angle_options)

        # サイズが1以下の場合は単純にコピー
        if size <= 1:
            o1.sequence = p1.sequence.copy()
            o2.sequence = p2.sequence.copy()
            o1.angles = p1.angles.copy()
            o2.angles = p2.angles.copy()
        else:
            # 順序交叉
            point1, point2 = sorted(random.sample(range(size), 2))
            o1.sequence = self._ordered_crossover(p1.sequence, p2.sequence, point1, point2)
            o2.sequence = self._ordered_crossover(p2.sequence, p1.sequence, point1, point2)

            # 角度の一点交叉
            cross_point = random.randint(1, size - 1)
            o1.angles = p1.angles[:cross_point] + p2.angles[cross_point:]
            o2.angles = p2.angles[:cross_point] + p1.angles[cross_point:]

        return o1, o2
    
    def _ordered_crossover(self, p1_seq, p2_seq, point1, point2):
        """順序交叉の実装"""
        size = len(p1_seq)
        offspring = [-1] * size
        offspring[point1:point2] = p1_seq[point1:point2]
        
        p2_filtered = [x for x in p2_seq if x not in offspring[point1:point2]]
        idx = 0
        for i in range(size):
            if offspring[i] == -1:
                offspring[i] = p2_filtered[idx]
                idx += 1
        
        return offspring
    
    def mutation(self, individual: Individual):
        """突然変異（逆位変異）"""
        if random.random() < self.mutation_rate and self.n_trees >= 2:
            point1, point2 = sorted(random.sample(range(self.n_trees), 2))
            individual.sequence[point1:point2+1] = list(reversed(individual.sequence[point1:point2+1]))

        for i in range(self.n_trees):
            if random.random() < self.mutation_rate:
                individual.angles[i] = random.choice(self.angle_options)
    
    def evolve(self):
        """一世代進化"""
        parents = self.selection()
        next_generation = []
        
        for i in range(self.elite_size):
            next_generation.append(parents[i].copy())
        
        while len(next_generation) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            o1, o2 = self.crossover(p1, p2)
            self.mutation(o1)
            self.mutation(o2)
            next_generation.append(o1)
            if len(next_generation) < self.population_size:
                next_generation.append(o2)
        
        self.population = next_generation[:self.population_size]
    
    def run(self, verbose: bool = True) -> Individual:
        """GAを実行"""
        start_time = time.time()
        
        self.initialize_population()
        print(f"初期集団生成完了 ({self.population_size}個体)")
        
        print("初期集団の評価中...")
        self.evaluate_population()
        
        self.best_individual = max(self.population, key=lambda x: x.fitness).copy()
        self.fitness_history.append(self.best_individual.fitness)
        
        print(f"初期最良解: コンテナサイズ = {self.best_individual.container_size:.4f}")
        print(f"           コンテナ面積 = {self.best_individual.container_size**2:.4f}")
        
        iterator = range(self.max_generations)
        if verbose:
            iterator = tqdm(iterator, desc="GA進化中")
        
        for gen in iterator:
            self.evolve()
            self.evaluate_population()
            
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            self.fitness_history.append(self.best_individual.fitness)
            
            if verbose:
                iterator.set_postfix({
                    'size': f'{self.best_individual.container_size:.4f}',
                    'area': f'{self.best_individual.container_size**2:.4f}',
                    'fitness': f'{self.best_individual.fitness:.2f}'
                })
        
        elapsed = time.time() - start_time
        print(f"\nGA完了 (実行時間: {elapsed:.2f}秒)")
        print(f"最終最良解: コンテナサイズ = {self.best_individual.container_size:.4f}")
        print(f"           コンテナ面積 = {self.best_individual.container_size**2:.4f}")
        
        return self.best_individual
    
    def visualize_solution(self, individual: Individual, save_path: Optional[str] = None):
        """解を可視化"""
        if individual.trees is None:
            print("可視化するツリーがありません")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # ツリーを描画
        for i, tree in enumerate(individual.trees):
            x, y = tree.polygon.exterior.xy
            if tree.use_scale:
                x_scaled = [xi / float(SCALE_FACTOR) for xi in x]
                y_scaled = [yi / float(SCALE_FACTOR) for yi in y]
            else:
                x_scaled = list(x)
                y_scaled = list(y)
            
            ax.plot(x_scaled, y_scaled, 'b-', linewidth=0.8)
            ax.fill(x_scaled, y_scaled, alpha=0.3, color='green')
        
        # 正方形コンテナを描画
        container_size = individual.container_size
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (0, 0),
            container_size,
            container_size,
            fill=False,
            edgecolor='red',
            linewidth=2.5,
            linestyle='--'
        )
        ax.add_patch(rect)
        
        ax.set_xlim(-0.5, container_size + 0.5)
        ax.set_ylim(-0.5, container_size + 0.5)
        ax.set_aspect('equal')
        ax.set_title(
            f'Christmas Tree Packing - Container Size: {container_size:.4f}\n'
            f'Container Area: {container_size**2:.4f} | Trees: {len(individual.trees)}',
            fontsize=14
        )
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可視化を保存: {save_path}")
        
        plt.show()
    
    def export_solution_csv(self, individual: Individual, filename: str):
        """解をCSV形式でエクスポート"""
        if individual.trees is None:
            print("エクスポートするツリーがありません")
            return
        
        def to_str(x: Decimal):
            return f"s{round(float(x), 6)}"
        
        rows = []
        n = len(individual.trees)
        
        for i, tree in enumerate(individual.trees):
            rows.append({
                "id": f"{n:03d}_{i}",
                "x": to_str(tree.center_x),
                "y": to_str(tree.center_y),
                "deg": to_str(tree.angle),
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"解をエクスポート: {filename}")


# テスト実行
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='改善版クリスマスツリーパッキング最適化')
    parser.add_argument('--n', type=int, default=10, help='ツリーの数 (デフォルト: 10)')
    parser.add_argument('--angle-step', type=int, default=5, help='回転角度の刻み幅 (デフォルト: 5度)')
    parser.add_argument('--population', type=int, default=50, help='GA個体数 (デフォルト: 50)')
    parser.add_argument('--generations', type=int, default=40, help='GA世代数 (デフォルト: 40)')
    parser.add_argument('--grid-step', type=float, default=0.02, help='グリッド探索刻み幅 (デフォルト: 0.02)')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='突然変異率 (デフォルト: 0.15)')
    parser.add_argument('--elite-ratio', type=float, default=0.1, help='エリート保存率 (デフォルト: 0.1)')
    parser.add_argument('--output-dir', type=str, default='./output_improved', help='出力ディレクトリ')

    args = parser.parse_args()

    print("=" * 60)
    print("改善版クリスマスツリーパッキングGA (Bottom-Left戦略)")
    print("=" * 60)

    N_TREES = args.n
    ANGLE_OPTIONS = list(range(0, 360, args.angle_step))

    print(f"\n設定:")
    print(f"  ツリー数: {N_TREES}")
    print(f"  回転角度: 0°-{360-args.angle_step}° ({args.angle_step}度刻み, {len(ANGLE_OPTIONS)}通り)")
    print(f"  個体数: {args.population}")
    print(f"  世代数: {args.generations}")
    print(f"  グリッドステップ: {args.grid_step}")
    print(f"  突然変異率: {args.mutation_rate}")
    print(f"  エリート率: {args.elite_ratio}")

    ga = ImprovedGeneticAlgorithm(
        n_trees=N_TREES,
        angle_options=ANGLE_OPTIONS,
        population_size=args.population,
        elite_ratio=args.elite_ratio,
        mutation_rate=args.mutation_rate,
        max_generations=args.generations,
        grid_step=args.grid_step,
        impact_factor=1000.0
    )

    print("\n" + "=" * 60)
    print("GA実行開始...")
    print("=" * 60)

    best_solution = ga.run(verbose=True)

    print("\n" + "=" * 60)
    print("最終結果:")
    print("=" * 60)
    print(f"  コンテナサイズ: {best_solution.container_size:.6f}")
    print(f"  コンテナ面積: {best_solution.container_size**2:.6f}")
    print(f"  適応度: {best_solution.fitness:.6f}")

    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV出力
    csv_path = f"{args.output_dir}/solution_n{N_TREES}.csv"
    ga.export_solution_csv(best_solution, csv_path)

    # 可視化
    solution_path = f"{args.output_dir}/solution_n{N_TREES}.png"
    print(f"\n可視化中...")
    ga.visualize_solution(best_solution, save_path=solution_path)

    print(f"\n配置図: {solution_path}")
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)