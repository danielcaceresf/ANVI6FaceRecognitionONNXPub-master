﻿#pragma checksum "C:\Lab_Api_Face\ANVI6FaceRecognitionONNXPub-master\App4\Altas.xaml" "{406ea660-64cf-4c82-b6f0-42d48172a799}" "84308BBEFFB0C8E972E9F38507E49D7C"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace App4
{
    partial class Altas : 
        global::Windows.UI.Xaml.Controls.Page, 
        global::Windows.UI.Xaml.Markup.IComponentConnector,
        global::Windows.UI.Xaml.Markup.IComponentConnector2
    {
        /// <summary>
        /// Connect()
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public void Connect(int connectionId, object target)
        {
            switch(connectionId)
            {
            case 2: // Altas.xaml line 68
                {
                    this.itemListView = (global::Windows.UI.Xaml.Controls.ListView)(target);
                    ((global::Windows.UI.Xaml.Controls.ListView)this.itemListView).ItemClick += this.itemListView_ItemClick;
                }
                break;
            case 3: // Altas.xaml line 47
                {
                    this.btnCrearPersona = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnCrearPersona).Click += this.Button_Click;
                }
                break;
            case 4: // Altas.xaml line 48
                {
                    this.personaNombre = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 5: // Altas.xaml line 49
                {
                    this.statusCreacion = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 6: // Altas.xaml line 50
                {
                    this.btnAgregarCara = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnAgregarCara).Click += this.ButtonAgregarCara_Click;
                }
                break;
            case 7: // Altas.xaml line 51
                {
                    this.urlImagen = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 8: // Altas.xaml line 52
                {
                    this.imagenAltas = (global::Windows.UI.Xaml.Controls.Image)(target);
                }
                break;
            case 9: // Altas.xaml line 53
                {
                    this.urlImagenStatus = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 10: // Altas.xaml line 54
                {
                    this.btnEntrenar = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnEntrenar).Click += this.ButtonEntrenar_Click;
                }
                break;
            case 11: // Altas.xaml line 55
                {
                    global::Windows.UI.Xaml.Controls.Button element11 = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)element11).Click += this.Button_Click_1;
                }
                break;
            case 12: // Altas.xaml line 56
                {
                    this.statusItem = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 13: // Altas.xaml line 57
                {
                    this.statusItemEliminado = (global::Windows.UI.Xaml.Controls.TextBox)(target);
                }
                break;
            case 14: // Altas.xaml line 59
                {
                    this.delItem = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.delItem).Click += this.EliminarUsuario;
                }
                break;
            case 15: // Altas.xaml line 63
                {
                    this.btnSubirImagen = (global::Windows.UI.Xaml.Controls.Button)(target);
                    ((global::Windows.UI.Xaml.Controls.Button)this.btnSubirImagen).Click += this.ButtonSubirImagen_Click;
                }
                break;
            default:
                break;
            }
            this._contentLoaded = true;
        }

        /// <summary>
        /// GetBindingConnector(int connectionId, object target)
        /// </summary>
        [global::System.CodeDom.Compiler.GeneratedCodeAttribute("Microsoft.Windows.UI.Xaml.Build.Tasks"," 10.0.17.0")]
        [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
        public global::Windows.UI.Xaml.Markup.IComponentConnector GetBindingConnector(int connectionId, object target)
        {
            global::Windows.UI.Xaml.Markup.IComponentConnector returnValue = null;
            return returnValue;
        }
    }
}

