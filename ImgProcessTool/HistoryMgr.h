#pragma once
#include <QStack>
#include <QImage>
#include <QSharedPointer>

class HistoryMgr
{
public:
	explicit HistoryMgr(int maxHistory = 30);

	// 保存当前状态（在修改图像之前调用）
	void pushState(const QImage& image);

	// 撤销：返回上一步的状态
	QImage undo();

	// 重做：返回下一步的状态
	QImage redo();

	// 查询是否可以撤销/重做
	bool canUndo() const { return m_undoStack.size() > 1; }
	bool canRedo() const { return !m_redoStack.isEmpty(); }

	// 获取当前状态（最近一次保存的状态）
	QImage currentState() const;

	// 清空所有历史
	void clear();

	// 设置最大历史步数（会清空现有历史）
	void setMaxHistory(int max);
	int maxHistory() const { return m_maxHistory; }

	// 获取当前历史步数
	int undoSteps() const { return m_undoStack.size() - 1; }
	int redoSteps() const { return m_redoStack.size(); }

	// 调试用：打印历史信息
	void debugPrint() const;

private:
	QStack<QSharedPointer<QImage>> m_undoStack;  // 撤销栈
	QStack<QSharedPointer<QImage>> m_redoStack;  // 重做栈
	int m_maxHistory;

	void trimHistory();  // 限制历史步数
	QSharedPointer<QImage> copyImage(const QImage& image) const;
};

