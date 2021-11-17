#include <Eigen/Core>
#include <Eigen/Dense> // 包含Eigen库里所有的函数的类
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
using namespace std;

void test01() {
  Eigen::MatrixXd m(2, 2); // MatrixXd 表示动态数组，初始化的时候指定行数和列数
  m(0, 0) = 2;
  m(0, 1) = 1.5;
  m(1, 0) = -1;
  m(1, 1) = m(0, 0) + m(0, 1);
  std::cout << m << std::endl;
}

// 矩阵和向量
void test02() {
  // 随机矩阵，值在（-1， 1）之间
  Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 3);
  // Constant(3, 3, 1.2) 初始化常量矩阵，值全部为1.2
  m = (m + Eigen::MatrixXd::Constant(3, 3, 1.2)) * 50;
  cout << m << endl;
  Eigen::VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v = " << endl << m * v << endl;
}

// 逗号初始化
void test03() {
  Eigen::RowVectorXd v1(3), v2(4), v3(7);
  v1 << 1, 2, 3;
  cout << v1 << endl;

  v2 << 1, 4, 9, 16;
  cout << v2 << endl;

  v3 << v1, v2;
  cout << v3 << endl;
}

// 初始化为0，1，单位矩阵
void test04() {
  Eigen::MatrixXd m0 = Eigen::MatrixXd::Random(3, 3); // 3×3的随机矩阵
  Eigen::MatrixXd m1 =
      Eigen::MatrixXd::Constant(3, 3, 2.4);         // 常量为2.4的3×3矩阵
  Eigen::MatrixXd m2 = Eigen::Matrix2d::Zero();     // 2×2的0矩阵
  Eigen::MatrixXd m3 = Eigen::Matrix3d::Ones();     // 3×3的1矩阵
  Eigen::MatrixXd m4 = Eigen::Matrix4d::Identity(); // 4×4的单位矩阵
  Eigen::Matrix3d m5;
  m5 << 1, 2, 3, 4, 5, 6, 7, 8, 9; // 自动填充3×3的矩阵
  cout << m0 << endl;
  cout << m1 << endl;
  cout << m2 << endl;
  cout << m3 << endl;
  cout << m4 << endl;
  cout << m5 << endl;

  Eigen::MatrixXf m6 = Eigen::MatrixXf::Ones(2, 3); // 2×3的0矩阵
  cout << m6 << endl;
  // 使用逗号初始化的临时变量必须用finished来获取矩阵对象
  m6 = (Eigen::MatrixXf(2, 2) << 0, 1, 2, 0).finished() * m6;
  cout << m6 << endl;
}

// 调整矩阵大小
// 动态矩阵可以随意调整矩阵大小
// 固定尺寸的矩阵无法调整大小
void test05() {
  Eigen::MatrixXd m(2, 5);
  m.resize(4, 3);                              // 调整大小
  cout << m.rows() << "x" << m.cols() << endl; // 行数、列数
  cout << m.size() << endl;                    //系数

  Eigen::VectorXd v(2);
  v.resize(5);
  cout << v.size() << endl;
  cout << v.rows() << "x" << v.cols() << endl;
}

// Matrix类
//三个必需参数：类型，行数，列数
// Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
void test06() {
  // Eigen 中自带的typedef简化名
  typedef Eigen::Matrix<float, 3, 1> Vector3f;
  typedef Eigen::Matrix<int, 1, 2> RowVector2i;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
  typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;
}

// Array类
//三个必需参数：类型，行数，列数
// Array<typename Scalar, int RowsAtCompileTime , int ColsAtCompileTime >
void test07() {
  typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> ArrayXXf;
  typedef Eigen::Array<double, Eigen::Dynamic, 1> ArrayXd;
  typedef Eigen::Array<int, 1, Eigen::Dynamic> RowArrayXi;
  typedef Eigen::Array<float, 3, 3> Array33f;
  typedef Eigen::Array<float, 4, 1> Array4f;
  typedef Eigen::Array<float, Eigen::Dynamic, 1> ArrayXf;
  typedef Eigen::Array<float, 3, 1> Array3f;
  typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> ArrayXXd;
  typedef Eigen::Array<double, 3, 3> Array33d;
}

// Array初始化，加减乘除操作
void test08() {
  Eigen::ArrayXXf a(3, 3);
  Eigen::ArrayXXf b(3, 3);
  a << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  b << 1, 2, 3, 1, 2, 3, 1, 2, 3;
  cout << a + b << endl;
  cout << a - b << endl;
  cout << a * b << endl;
  cout << a / b << endl;
}

// Array 绝对值、平方根、对应元素最小值
void test09() {
  Eigen::ArrayXXf a = Eigen::ArrayXXf::Random(2, 2);
  a *= 2;
  cout << a << endl;
  cout << a.abs() << endl;
  cout << a.abs().sqrt() << endl;
  cout << a.min(a.abs().sqrt()) << endl;
}

// Array和Matrix相互转换
void test10() {
  Eigen::Array44f a1, a2;
  Eigen::Matrix4f m1, m2;
  m1 = a1 * a2;          // 系数乘积，从数组到矩阵的隐式转换
  a1 = m1 * m2;          // 矩阵乘积，从矩阵到数组的隐式转换
  a2 = a1 + m1.array();  // 必须显式转换后才能相加
  m2 = a1.matrix() + m1; // 必须显式转换后才能相加

  // m1a是m1.array() 的别名，共享相同的系数
  Eigen::ArrayWrapper<Eigen::Matrix4f> m1a(m1);
  Eigen::MatrixWrapper<Eigen::Array44f> a1m(a1);
  cout << a1 << endl << endl;
  cout << a2 << endl << endl;
  cout << m1 << endl << endl;
  cout << m2 << endl << endl;
  cout << m1a << endl << endl;
  cout << a1m << endl;
}

// 矩阵转置、共轭、共轭转置
void test11() {
  Eigen::MatrixXcf a = Eigen::MatrixXcf::Random(2, 2);
  cout << a << endl << endl;
  cout << a.transpose() << endl << endl;
  cout << a.conjugate() << endl << endl;
  cout << a.adjoint() << endl;
}

// 转置需要注意的问题
// 别名问题，在debug模式下当assertions没有禁止时，这种问题会被自动检测到。
// 要避免错误，可以使用in-place转置。类似的还有adjointInPlace()。
void test12() {
  Eigen::Matrix2i a;
  a << 1, 2, 3, 4;
  cout << a << endl << endl;
  // a=a.transpose();  // 错误
  a.transposeInPlace();
  cout << a << endl;
}

// 点积和叉积
// 叉积仅仅用于尺寸为3的向量！点积可以用于任意尺寸的向量
// 当使用复数时，Eigen的点积操作是第一个变量为共轭线性的，第二个为线性的
void test13() {
  Eigen::Vector3d v1(1, 2, 3);
  Eigen::Vector3d v2(0, 1, 2);
  cout << v1.dot(v2) << endl << endl; // 点积
  double dp = v1.adjoint() * v2;
  cout << dp << endl << endl;
  cout << v1.cross(v2) << endl; //叉积
}

// 矩阵的基础运算，求和，平均
void test14() {
  Eigen::Matrix2d m;
  m << 1, 2, 3, 4;
  cout << m.sum() << endl;            // 和
  cout << m.prod() << endl;           // 乘积
  cout << m.mean() << endl;           // 平均
  cout << m.minCoeff() << endl;       // 最小值
  cout << m.maxCoeff() << endl;       //最大值
  cout << m.trace() << endl;          // 矩阵的迹，对角线之和
  cout << m.diagonal().sum() << endl; // 矩阵的迹
}

// minCoeff和maxCoeff函数也可以返回相应的元素的位置信息
void test15() {
  Eigen::Matrix3f m = Eigen::Matrix3f::Random();
  std::ptrdiff_t i, j;
  float minOfM = m.minCoeff(&i, &j);
  cout << m << endl << endl;
  cout << minOfM << " " << i << " " << j << endl;

  Eigen::RowVector4i v = Eigen::RowVector4i::Random();
  int maxOfV = v.maxCoeff(&i);
  cout << v << endl << endl;
  cout << maxOfV << " " << i << endl;
}

// 块操作
void test16() {
  Eigen::MatrixXf m(4, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      m(i, j) = j + 1 + i * 4;
    }
  }

  cout << "m:" << endl << m << endl;
  // m.block<i,j>(a,b)表示从第（a+1）行(b+1)列开始，截i行j列
  cout << m.block<2, 2>(1, 1) << endl;

  for (int i = 1; i <= 3; i++) {
    cout << "block size" << i << "x" << i << endl;
    // m.block(a,b,i,j)表示从第(a+1)行(b+1)列开始，截i行，j列
    cout << m.block(0, 0, i, i) << endl << endl;
  }
}

// 块赋值
void test17() {
  Eigen::Array22f m;
  m << 1, 2, 3, 4;
  Eigen::Array44f a = Eigen::Array44f::Constant(0.6);
  cout << "a:" << endl << a << endl << endl;
  a.block<2, 2>(1, 1) = m;
  cout << "a:" << endl << a << endl << endl;
  a.block(0, 0, 2, 3) = a.block(2, 1, 2, 3);
  cout << "a:" << endl << a << endl << endl;
}

// 行和列
void test18() {
  Eigen::MatrixXf m(4, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      m(i, j) = j + 1 + i * 4;
    }
  }

  cout << "m:" << endl << m << endl;
  cout << "第2行:" << m.row(1) << endl;
  m.col(2) += 3 * m.col(0);
  cout << "m:" << m << endl;
}

// 边角操作，左上角、右上角。。。
void test19() {
  Eigen::Matrix4f m;
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  cout << "m.leftCols(2)" << endl << m.leftCols(2) << endl << endl;
  cout << "m.bootomRows<2>()" << endl << m.bottomRows<2>() << endl << endl;
  m.topLeftCorner(1, 3) = m.bottomRightCorner(3, 1).transpose();
  cout << "m:" << endl << m << endl;
}

// 对于向量的块操作
void test20() {
  Eigen::ArrayXf v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3)=" << endl << v.head(3) << endl << endl;
  cout << "v.tail(3)=" << endl << v.tail<3>() << endl << endl;
  v.segment(1, 4) *= 2;
  cout << v << endl;
}

// 范数计算
void test21() {
  Eigen::VectorXf v(2);
  Eigen::MatrixXf m(2, 2), n(2, 2);
  v << -1, 2;
  m << 1, -2, -3, 4;
  cout << v.squaredNorm() << endl; // 平方范数，向量自身做点积，所有元素平方和
  cout << v.norm() << endl;      // 2范数，所有元素平方和的平方根
  cout << v.lpNorm<1>() << endl; // 绝对值和
  cout << v.lpNorm<Eigen::Infinity>() << endl;
  cout << m.squaredNorm() << endl;
  cout << m.norm() << endl;
  cout << m.lpNorm<1>() << endl;
  cout << m.lpNorm<Eigen::Infinity>() << endl;
}

// 矩阵的1范数和无穷范数
void test22() {
  Eigen::MatrixXf m(2, 2);
  m << 1, -2, -3, 4;
  // 1范数，绝对值之和
  cout << m.cwiseAbs().colwise().sum().maxCoeff()
       << "==" << m.colwise().lpNorm<1>().maxCoeff() << endl; //列绝对值之和最大

  //  无穷范数，绝对值最大的
  cout << m.cwiseAbs().rowwise().sum().maxCoeff()
       << "==" << m.rowwise().lpNorm<1>().maxCoeff()
       << endl; // 行绝对值之和最大
}

// 布尔规约
// all() 所有元素为真
// any() 有一个元素为真
// count() 元素为真的个数
void test23() {
  Eigen::ArrayXXf a(2, 2);
  a << 1, 2, 3, 4;
  cout << (a > 0).all() << endl;
  cout << (a > 0).any() << endl;
  cout << (a > 0).count() << endl;

  cout << (a > 2).all() << endl;
  cout << (a > 2).any() << endl;
  cout << (a > 2).count() << endl;
}

// 当需要获得元素在矩阵或数组中的位置时使用迭代
void test24() {
  Eigen::MatrixXf m(2, 2);
  m << 1, 2, 3, 4;

  // 获得最大值的位置
  Eigen::MatrixXf::Index maxRow, maxCol;
  float max = m.maxCoeff(&maxRow, &maxCol);

  // 获得最小值的位置
  Eigen::MatrixXf::Index minRow, minCol;
  float min = m.minCoeff(&minRow, &minCol);

  cout << "最大值：" << max << "在" << maxRow << "," << maxCol << endl;
  cout << "最小值：" << min << "在" << minRow << "," << minCol << endl;
}

// 部分规约
// 对矩阵或数组按行或列进行操作
void test25() {
  Eigen::MatrixXf m(2, 4);
  m << 1, 2, 6, 9, 3, 1, 7, 2;

  // 得到矩阵每一列的最大值并存入一个行向量中
  cout << "每一列的最大值：" << endl << m.colwise().maxCoeff() << endl;
  // 得到矩阵每一列的最大值并存入一个列向量中
  cout << "每一行的最大值：" << endl << m.rowwise().maxCoeff() << endl;
}

// 部分规约和其他操作的结合
// 得到矩阵中元素和最大的一列
void test26() {
  Eigen::MatrixXf m(2, 4);
  m << 1, 2, 6, 9, 3, 1, 7, 2;
  Eigen::MatrixXf::Index maxIndex;
  float maxNorm = m.colwise().sum().maxCoeff(&maxIndex);

  cout << "最大和的位置：" << maxIndex << endl;
  cout << "对应的向量：" << endl << m.col(maxIndex) << endl;
  cout << "和为：" << maxNorm << endl;
}

// 广播机制
// 广播通过对向量在一个方向上的复制，将向量解释成矩阵
void test27() {
  Eigen::MatrixXf m(2, 4);
  m << 1, 2, 6, 9, 3, 1, 7, 2;

  Eigen::VectorXf v(2);
  v << 0, 1;

  // 将一个列向量加到矩阵的每一列中
  m.colwise() += v;
  cout << m << endl;
}

// 在矩阵中找到和给定向量最接近的一列
void test28() {
  Eigen::MatrixXf m(2, 4);
  Eigen::VectorXf v(2);
  m << 1, 23, 6, 9, 3, 11, 7, 2;
  v << 2, 3;

  Eigen::MatrixXf::Index index;
  // 使用欧式距离，找最近邻
  // 按列计算平方范数
  (m.colwise() - v).colwise().squaredNorm().minCoeff(&index);
  cout << "最近的列：" << index << ":" << endl << m.col(index) << endl;
}

// 几何模块
// 空间旋转/平移
// 实际中物体不光有旋转，还有平移运动，如果用t表示平移向量，
// 那么R*p+t可以描述刚体p的旋转加平移运动，然而当连续多次运动时
// 整个表达式将会变得非常复杂，比如R1*(R*p+t)+t1描述连续两次的运动，
// 因此为了简化书写形式引入齐次坐标的概念，将坐标扩充到4维，
// 将旋转矩阵和平移向量写入一个4x4的变换矩阵中，简化了连续运动公式的形式，
// 但是结果是16个参数描述一个6自由度的运动，更加冗余了。
// 在旋转向量的后面增加3维代表平移向量，即用6维的旋转向量描述旋转和平移运动，
// 看起来比较紧凑了，但是像欧拉角一样也会遇到万向锁问题，导致奇异性；
// 最终即不冗余又紧凑又没有万向锁问题的解决方案是使用四元数描述旋转问题，
// 这也是很多飞控代码中用到的方案。
void test29() {
  // 旋转矩阵直接使用Matrix3d
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix.setIdentity();

  // 旋转向量 由旋转轴和旋转角度组成
  Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
  cout.precision(3); // 保留三位小数

  cout << "旋转向量：角度是：" << rotation_vector.angle() * (180 / M_PI)
       << "坐标轴是：" << rotation_vector.axis().transpose() << endl;
  cout << "旋转矩阵为:" << rotation_vector.matrix() << endl;

  rotation_matrix = rotation_vector.toRotationMatrix();
  //  v是待旋转的向量，或空间中一个刚体的位置
  Eigen::Vector3d v(1, 0, 0);
  Eigen::Vector3d v_rotated = rotation_vector * v;
  cout << "(1,0,0)旋转之后：" << v_rotated.transpose() << endl;
  v_rotated = rotation_matrix * v;
  cout << "(1,0,0)旋转之后：" << v_rotated.transpose() << endl;

  // 欧拉角按ZYX的顺序，由旋转矩阵直接转换成欧拉角
  Eigen::Vector3d euler_angles =
      rotation_matrix.eulerAngles(2, 1, 0); // 2代表Z轴，1代表Y轴，0代表X轴
  cout << "yaw pitch roll:" << euler_angles.transpose() * (180 / M_PI) << endl;

  // 变换4×4矩阵
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(rotation_vector);
  // T.rotate(rotation_matrix); // 这样也可以，相当于由旋转矩阵构造变换矩阵

  // 设置平移量
  T.pretranslate(Eigen::Vector3d(0, 0, 3));
  cout << "旋转矩阵为：" << T.matrix() << endl;

  // 用变换矩阵进行坐标变换
  Eigen::Vector3d v_transformed = T * v;
  cout << "v变换之后：" << v_transformed.transpose() << endl;

  // 由旋转向量构造四元数
  Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
  cout << "四元数：" << q.coeffs() << endl;

  // 由旋转矩阵构造四元数
  q = Eigen::Quaterniond(rotation_matrix);
  cout << "四元数：" << q.coeffs() << endl;

  v_rotated = q * v;
  cout << "(1,0,0)旋转之后" << v_rotated.transpose() << endl;
}

int main(int argc, char const *argv[]) {
  //   test01();
  //   test02();
  //   test03();
  //   test04();
  //   test05();
  //   test08();
  //   test09();
  //   test10();
  //   test11();
  //   test12();
  //   test13();
  //   test14();
  // test15();
  // test16();
  // test17();
  // test18();
  // test19();
  // test20();
  // test21();
  // test22();
  // test23();
  // test24();
  // test25();
  // test26();
  // test27();
  // test28();
  test29();

  return 0;
}
