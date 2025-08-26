@echo off
echo Activating Conda environment...
call conda activate myprojectenv

echo Running main_image_gradio.py...
python main_image_gradio.py

echo Deactivating Conda environment...
call conda deactivate

echo Done!
echo.
echo [DEBUG MODE] Press any key to exit...
pause >nul
