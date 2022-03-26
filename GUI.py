import tkinter as tk
import tkinter.filedialog
from Dlib import dlib_eval
import sys
sys.path.append('SAN')
import san_eval


# create window
window = tk.Tk()
# set title
window.title('Lip Motion Detection')
# set window size
sw = window.winfo_screenwidth()
sh = window.winfo_screenheight()
ww = 650
wh = 500
x = int((sw-ww) / 2)
y = int((sh-wh) / 2)
window.geometry(f"{ww}x{wh}+{x}+{y}")


# select processing method
method_type = tk.StringVar()
label_method = tk.Label(window, font=('Arial', 14), text='1. Please select the method:')
label_method.place(x=10, y=10)
# method selection function
def set_Dlib():
    r_btn_input_3['state'] = 'normal'
def set_SAN():
    r_btn_input_3['state'] = 'disabled'
    if input_type.get() == 'Camera':
        input_type.set('Image')
        set_Image()
# radiobutton_method
method_type.set('Dlib')
r_btn_method_1 = tk.Radiobutton(window, font=('Arial', 14), text='Dlib', variable=method_type, value='Dlib', command=set_Dlib)
r_btn_method_1.place(x=100, y=40)
r_btn_method_2 = tk.Radiobutton(window, font=('Arial', 14), text='SAN', variable=method_type, value='SAN', command=set_SAN)
r_btn_method_2.place(x=250, y=40)


# select input type
input_type = tk.StringVar()
label_input = tk.Label(window, font=('Arial', 14), text='2. Please select the input type:')
label_input.place(x=10, y=100)
label_input_notice = tk.Label(window, font=('Arial', 14), text='  (NOTICE that SAN now does NOT support camera input.)')
label_input_notice.place(x=10, y=125)
# input selection function
def set_Image():
    entry_upload_path['state'] = 'normal'
    btn_upload['state'] = 'normal'
def set_Video():
    entry_upload_path['state'] = 'normal'
    btn_upload['state'] = 'normal'
def set_Camera():
    entry_upload_path['state'] = 'disabled'
    btn_upload['state'] = 'disabled'
# radiobutton_input
input_type.set('Image')
r_btn_input_1 = tk.Radiobutton(window, font=('Arial', 14), text='Image', variable=input_type, value='Image', command=set_Image)
r_btn_input_1.place(x=100, y=155)
r_btn_input_2 = tk.Radiobutton(window, font=('Arial', 14), text='Video', variable=input_type, value='Video', command=set_Video)
r_btn_input_2.place(x=250, y=155)
r_btn_input_3 = tk.Radiobutton(window, font=('Arial', 14), text='Camera', variable=input_type, value='Camera', command=set_Camera)
r_btn_input_3.place(x=400, y=155)


# upload media
def selectPath():
    path_ = tkinter.filedialog.askopenfilename()
    # path_ = path_.replace("/","\\\\")
    upload_path.set(path_)
upload_path = tk.StringVar()
label_upload = tk.Label(window, text = "3. File Path:", font=('Arial', 14))
label_upload.place(x=10, y=225)
entry_upload_path = tk.Entry(window, font=('Arial', 14), textvariable = upload_path)
entry_upload_path.place(x=120, y=227, width=350)
btn_upload = tk.Button(window, text = "Select File", font=('Arial', 14), width=10, command = selectPath)
btn_upload.place(x=480, y=220)


# set saved path
tk.Label(window, text='4. Saved Path:', font=('Arial', 14)).place(x=10, y=315)
saved_path = tk.StringVar()
saved_path.set('processed_media/')
entry_saved_path = tk.Entry(window, textvariable=saved_path, font=('Arial', 14))
entry_saved_path.place(x=140, y=317, width=350)


# process button
def precess():
    if method_type.get() == 'Dlib':
        args = dlib_eval.Dlib_Args(input_type=input_type.get(), input=upload_path.get(), save_path=saved_path.get())
        dlib_eval.execute(args)
    else:
        args = san_eval.SAN_Args(input_type=input_type.get(), input=upload_path.get(), save_path=saved_path.get())
        args.execute()
    topw, toph = 210, 100
    ctpx = int(sw / 2 - topw / 2)
    ctpy = int(sh / 2 - toph / 2)
    top = tk.Toplevel(window)
    top.geometry(f"{topw}x{toph}+{ctpx}+{ctpy}")
    top.title('Notice')
    finish_top = tk.Label(top, font=('Arial', 14), text=' Process completed. ')
    finish_top.place(x=10, y=10)
    btn_ok = tk.Button(top, font=('Arial', 14), text="OK", width=8, command=top.destroy)
    btn_ok.place(x=55, y=50)

btn_process = tk.Button(window, font=('Arial', 14), text="Process", width=10, command=precess)
btn_process.place(x=120, y=370)
# quit button
btn_quit = tk.Button(window, font=('Arial', 14), text="Quit", width=10, command=window.quit)
btn_quit.place(x=280, y=370)


window.mainloop()