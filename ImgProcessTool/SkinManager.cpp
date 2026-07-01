#include "SkinManager.h"
#include <QFile>
#include <QApplication>
#include <QDebug>
#include <QSettings>

SkinManager* SkinManager::m_instance = nullptr;

SkinManager* SkinManager::instance()
{
	if (!m_instance)
	{
		m_instance = new SkinManager();
	}
	return m_instance;
}

void SkinManager::loadSkins()
{
	// 注册所有皮肤
	m_skinMap["black"] = ":/qss/black.qss";
	m_skinMap["white"] = ":/qss/white.qss";
	m_skinMap["green"] = ":/qss/green.qss";

	// 映射枚举
	m_skinTypeMap["black"] = Black;
	m_skinTypeMap["white"] = White;
	m_skinTypeMap["green"] = Green;
	m_skinTypeMap["Default"] = None;

	// 从配置中读取上次使用的皮肤（可选）
	QSettings settings;
	QString lastSkin = settings.value("skin", "white").toString();
	if (m_skinMap.contains(lastSkin)) {
		setSkin(lastSkin);
	}
	else {
		setSkin(White);  // 默认白色
	}
}

bool SkinManager::setSkin(SkinType type)
{
	// 根据枚举找到对应的皮肤名称
	for (auto it = m_skinTypeMap.begin(); it != m_skinTypeMap.end(); ++it) {
		if (it.value() == type) {
			return setSkin(it.key());
		}
	}
	return false;
}

bool SkinManager::setSkin(const QString& skinName)
{
	if (!m_skinMap.contains(skinName)) {
		qDebug() << "皮肤不存在:" << skinName;
		return false;
	}

	QString qssPath = m_skinMap[skinName];

	// 加载 QSS 文件
	QFile file(qssPath);
	if (!file.open(QFile::ReadOnly)) {
		qDebug() << "无法加载 QSS 文件:" << qssPath;
		return false;
	}

	QString styleSheet = QLatin1String(file.readAll());
	qApp->setStyleSheet(styleSheet);

	// 更新当前状态
	m_currentSkin = skinName;
	m_currentSkinType = m_skinTypeMap.value(skinName, None);

	// 保存配置
	QSettings().setValue("skin", skinName);

	// 发送信号
	emit skinChanged(skinName);

	qDebug() << "? 切换到皮肤:" << skinName;
	return true;
}

SkinManager::SkinManager(QObject *parent)
	:QObject(parent)
	,m_currentSkinType(White)
{
	loadSkins();
}


SkinManager::~SkinManager()
{
}
