function change_checkbox(elem) {
    var checkbox = elem.parentElement.querySelector('.custom_checkbox_input');
    checkbox.checked = checkbox.checked ? false : true;
}

function start_script(elem, event) {
    var data = null, test = null, cls = null, cls_name = null, bagging = false;

    var command = "python3 ../classifier.py";
    // command = "";

    bagging = document.querySelector('#bagging').checked;

    if (document.querySelector('#data').files[0]) {
        command = `${command} -d "${ document.querySelector('#data').files[0].path }"`;
        
        if (document.querySelector('#test').files[0]) {
            command = `${command} -t "${ document.querySelector('#test').files[0].path }"`;
        }
    }

    if (bagging)
        command = `${command} -b`;

    cls = document.querySelector('#cls');
    cls_name = cls.options[cls.options.selectedIndex].innerHTML.trim();
    command = `${command} -c ${cls_name}`;

    eel.start_script(command)
}
