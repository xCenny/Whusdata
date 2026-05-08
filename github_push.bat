@echo off
echo ========================================================
echo Degisiklikler GitHub'a gonderiliyor...
echo ========================================================

git add .
git commit -m "fix: LLM timeout eklendi, sonsuz dongu sorunu cozuldu ve tag alani sisteme entegre edildi"
git push

echo.
echo ========================================================
echo Islem tamamlandi! Bu pencereyi kapatabilirsiniz.
echo ========================================================
pause
