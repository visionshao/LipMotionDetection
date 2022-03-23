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
window.geometry('650x500')


# select processing method
method_type = tk.StringVar()
label_method = tk.Label(window, font=('Arial', 14), text='1. Please select the method:')
label_method.place(x=10, y=10)
# method selection function
def set_Dlib():
    r_btn_input_2['state'] = 'normal'
    # r_btn_input_3['state'] = 'normal'
def set_SAN():
    input_type.set('image')
    r_btn_input_2['state'] = 'disabled'
    # r_btn_input_3['state'] = 'disabled'
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
label_input_notice = tk.Label(window, font=('Arial', 14), text='  (NOTICE that SAN now only supports image input.)')
label_input_notice.place(x=10, y=125)
# radiobutton_input
input_type.set('image')
r_btn_input_1 = tk.Radiobutton(window, font=('Arial', 14), text='Image', variable=input_type, value='image')
r_btn_input_1.place(x=100, y=155)
r_btn_input_2 = tk.Radiobutton(window, font=('Arial', 14), text='Video', variable=input_type, value='video')
r_btn_input_2.place(x=250, y=155)
# r_btn_input_3 = tk.Radiobutton(window, font=('Arial', 14), text='Camera', variable=input_type, value='camera')
# r_btn_input_3.place(x=400, y=155)


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
var_saved_path = tk.StringVar()
var_saved_path.set('processed_media/')
entry_saved_path = tk.Entry(window, textvariable=var_saved_path, font=('Arial', 14))
entry_saved_path.place(x=140, y=317, width=350)


# process button
def precess():
    if method_type.get() == 'Dlib':
        args = dlib_eval.Dlib_Args(input_type=input_type.get(), input=upload_path.get(), save_path=var_saved_path.get())
        dlib_eval.execute(args)
    else:
        args = san_eval.SAN_Args(image=upload_path.get(), save_path=var_saved_path.get())

btn_process = tk.Button(window, font=('Arial', 14), text="Process", width=10, command=precess)
btn_process.place(x=120, y=370)
# quit button
btn_quit = tk.Button(window, font=('Arial', 14), text="Quit", width=10, command=window.quit)
btn_quit.place(x=280, y=370)


window.mainloop()