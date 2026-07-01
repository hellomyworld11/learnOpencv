#pragma once
#include <QObject>
#include <QMap>
#include <QString>

class SkinManager : public QObject
{
	Q_OBJECT
public:
	enum SkinType {
		None, Black, White, Green
	};

	static SkinManager* instance();

	void loadSkins();

	bool setSkin(SkinType type);

	bool setSkin(const QString& skinName);

	QString currentSkinName() {return m_currentSkin;}

	SkinType currentSkinType() { return m_currentSkinType; }

	QStringList  skinNames() { return m_skinMap.keys(); }
signals:
	void skinChanged(const QString &skinName);

private:
	SkinManager(QObject* parent = nullptr);

	~SkinManager();
	
	static SkinManager* m_instance;

	QMap<QString, QString> m_skinMap;
	QMap<QString, SkinType> m_skinTypeMap;
	QString m_currentSkin;
	SkinType m_currentSkinType;
};

