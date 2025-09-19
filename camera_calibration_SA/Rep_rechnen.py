import cv2
import numpy as np
import pandas as pd

# شناسه‌های دوربین‌ها
cam_ids = ['000068700312', '000325420812', '000365930112', '000368930112', '000409930112', '000436120812']

# مقادیر خطای بازپروژکشن که از نتایج دریافتی به دست آمده است برای هر کالیبراسیون استریو
reprojection_errors = {
    '000436120812 & 000325420812': [0.10535945546671358, 0.12427311724933857, 0.13135294981051376, 0.13322896686198935, 0.0645104854553071, 0.11259323744752338, 0.10671907089365412, 0.09213070995159159],
    '000436120812 & 000068700312': [0.08832907724115759, 0.12751095222571443, 0.16341647667349746, 0.11327767761579527, 0.09710458482320715, 0.17276684039275444, 0.10280974067466574]
}



# محاسبه میانگین خطای بازپروژکشن برای هر کالیبراسیون استریو
mean_reprojection_errors = []
for (cam1, cam2), errors in reprojection_errors.items():
    mean_error = np.mean(errors)
    mean_reprojection_errors.append({'Camera Pair': f'{cam1} & {cam2}', 'Mean Reprojection Error': mean_error})

# ساخت یک DataFrame برای نمایش مقادیر به صورت جدول
calibration_df = pd.DataFrame(mean_reprojection_errors)

# محاسبه میانگین کلی خطاهای بازپروژکشن
overall_mean_error = calibration_df['Mean Reprojection Error'].mean()

# افزودن میانگین کلی به جدول
overall_mean_df = pd.DataFrame([{'Camera Pair': 'Overall Mean', 'Mean Reprojection Error': overall_mean_error}])
calibration_df = pd.concat([calibration_df, overall_mean_df], ignore_index=True)

# نمایش جدول
print(calibration_df)

# ذخیره جدول به صورت فایل CSV (اختیاری)
#calibration_df.to_csv('calibration_results.csv', index=False)