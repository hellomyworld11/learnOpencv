#include "HistoryMgr.h"
#include <QDebug>

HistoryMgr::HistoryMgr(int maxHistory)
	: m_maxHistory(qMax(1, maxHistory))
{
}

QSharedPointer<QImage> HistoryMgr::copyImage(const QImage& image) const {
	// QImage 的拷贝构造是深拷贝，但使用 shared_ptr 方便内存管理
	return QSharedPointer<QImage>::create(image);
}

void HistoryMgr::pushState(const QImage& image) {
	// 如果当前栈顶与要保存的图像相同，则跳过（避免重复）
	if (!m_undoStack.isEmpty() && *m_undoStack.top() == image) {
		qDebug() << "History: Skipping duplicate state";
		return;
	}

	// 当执行新操作时，清空重做栈（因为历史分支被改变了）
	if (!m_redoStack.isEmpty()) {
		qDebug() << "History: Clearing redo stack (" << m_redoStack.size() << "items)";
		m_redoStack.clear();
	}

	// 保存新状态
	m_undoStack.push(copyImage(image));

	// 限制历史步数
	trimHistory();

	qDebug() << "History: Pushed new state. Undo size:" << m_undoStack.size()
		<< "Redo size:" << m_redoStack.size();
}

QImage HistoryMgr::undo() {
	if (!canUndo()) {
		qDebug() << "History: Cannot undo";
		return QImage();
	}

	// 将当前状态移到重做栈
	m_redoStack.push(m_undoStack.pop());

	qDebug() << "History: Undo performed. Undo size:" << m_undoStack.size()
		<< "Redo size:" << m_redoStack.size();

	// 返回上一步的状态（深拷贝，避免外部修改影响历史）
	return m_undoStack.top()->copy();
}

QImage HistoryMgr::redo() {
	if (!canRedo()) {
		qDebug() << "History: Cannot redo";
		return QImage();
	}

	// 将重做栈顶的状态移回撤销栈
	m_undoStack.push(m_redoStack.pop());

	qDebug() << "History: Redo performed. Undo size:" << m_undoStack.size()
		<< "Redo size:" << m_redoStack.size();

	// 返回重做后的状态
	return m_undoStack.top()->copy();
}

QImage HistoryMgr::currentState() const {
	if (m_undoStack.isEmpty()) {
		return QImage();
	}
	return m_undoStack.top()->copy();
}

void HistoryMgr::clear() {
	m_undoStack.clear();
	m_redoStack.clear();
	qDebug() << "History: Cleared all history";
}

void HistoryMgr::setMaxHistory(int max) {
	m_maxHistory = qMax(1, max);
	trimHistory();
}

void HistoryMgr::trimHistory() {
	// 保持撤销栈的大小不超过最大历史步数
	// 注意：栈底是最早的历史，需要保留至少一个状态（当前状态）
	while (m_undoStack.size() > m_maxHistory + 1) {
		// QStack 没有 removeFirst，需要转换处理
		QStack<QSharedPointer<QImage>> temp;
		while (m_undoStack.size() > 1) {
			temp.push(m_undoStack.pop());
		}
		// 移除最旧的（栈底）
		if (!temp.isEmpty()) {
			temp.pop();
		}
		// 重新构建栈
		while (!temp.isEmpty()) {
			m_undoStack.push(temp.pop());
		}
	}
}

void HistoryMgr::debugPrint() const {
	qDebug() << "=== History Manager Debug ===";
	qDebug() << "Max History:" << m_maxHistory;
	qDebug() << "Undo Stack Size:" << m_undoStack.size();
	qDebug() << "Redo Stack Size:" << m_redoStack.size();
	qDebug() << "Can Undo:" << canUndo();
	qDebug() << "Can Redo:" << canRedo();
	if (!m_undoStack.isEmpty()) {
		qDebug() << "Current Image Size:" << m_undoStack.top()->size();
		qDebug() << "Current Image Format:" << m_undoStack.top()->format();
	}
	qDebug() << "============================";
}