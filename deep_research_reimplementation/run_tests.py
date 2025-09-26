"""
通义深度研究复现 - 测试运行器
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("通义深度研究复现 - 运行所有测试")
    print("=" * 60)

    # 发现并运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.join(project_root, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")

    return success_rate == 100.0

def run_specific_test(test_name):
    """运行特定测试"""
    print(f"运行测试: {test_name}")

    try:
        suite = unittest.TestLoader().loadTestsFromName(test_name)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"运行测试失败: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行特定测试
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # 运行所有测试
        success = run_all_tests()

    # 返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()