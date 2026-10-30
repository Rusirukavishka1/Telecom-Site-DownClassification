# GitHub එකට upload කිරීම

මෙය upload කිරීමට සකස් කළ folder එකකි. GitHub වෙත තවම publish කර නැත.

## 1. ZIP එක extract කරන්න

`telecom-site-down-analysis` folder එක තුළ `app.py`, `README.md`, `.gitignore`, `data` සහ `results` තිබිය යුතුයි. ZIP file එක repository එකට upload නොකර, extract කළ project files commit කරන්න.

## 2. GitHub හි repository එකක් සාදන්න

GitHub account එකට sign in වී අලුත් empty repository එකක් සාදන්න.

- Suggested name: `telecom-site-down-analysis`
- Description: `Machine learning classification of recorded telecom site-down status with a Streamlit dashboard.`
- Visibility ඔබගේ අවශ්‍යතාව අනුව තෝරන්න.
- Initial README, .gitignore හෝ license auto-create නොකරන්න; මේ folder එකේ README සහ .gitignore දැනටමත් ඇත.
- Software license එකක් මේ package එකෙන් තෝරා දී නැත. Dataset redistribution rights තහවුරු කිරීම repository ownerගේ වගකීමයි.

## 3. Git භාවිතයෙන් upload කරන්න

Git for Windows install වී තිබිය යුතුයි. Anaconda Prompt හෝ Git Bash එකේ ඔබ extract කළ සැබෑ path එකට යන්න:

```bat
cd /d "C:\Users\USER\Desktop\telecom-site-down-analysis"
git --version
git init
git branch -M main
git add .
git status
git commit -m "Add telecom site-down classification project"
```

Git author details ඉල්ලුවොත් ඔබගේ නම සහ GitHub verified email එක project එක සඳහා configure කර commit එක නැවත කරන්න:

```bat
git config user.name "YOUR NAME"
git config user.email "YOUR VERIFIED EMAIL"
git commit -m "Add telecom site-down classification project"
```

`YOUR_USERNAME` ඔබගේ GitHub username එකෙන් replace කර, ඔබ සාදාගත් repository URL එක භාවිත කරන්න:

```bat
git remote add origin https://github.com/YOUR_USERNAME/telecom-site-down-analysis.git
git push -u origin main
```

Git විසින් ලබාදෙන authentication flow එක සම්පූර්ණ කරන්න. Password/token එක source code හෝ README තුළ දමන්න එපා.

## 4. Upload වූ පසු

Repository එකේ README, notebook, data සහ results තිබෙන බව බලන්න. Actions workflow එක tests run කරයි; remote run එකක් මෙහි සිට තහවුරු කර නැත.

පසුව වෙනස්කම් upload කිරීමට:

```bat
git add .
git commit -m "Update dashboard"
git push
```

App එක run කිරීමට `SETUP.md` බලන්න. Review & limitations navigation tab එක ඉවත් කර ඇති version එක මේ package එකේ ඇත. Model results සහ notebook analytical logic වෙනස් කර නැත.
